#!/usr/bin/env python3
"""Build a PointMixer ScanNet++-compatible one-room dataset from CloudCompare TXT.

The input TXT is expected to contain a header like:
//X Y Z R G B label Original_cloud_index Classification

Only source classes listed in a mapping TSV are converted. Empty scannetpp_name
entries are written as ignore_label. The output keeps ScanNet++ top100 class
indices so checkpoints trained with 100 classes remain loadable.
"""

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import torch


STUFF_NAMES = {"wall", "floor", "ceiling"}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-txt", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-meta", required=True,
                        help="Existing ScanNet++ top100 meta directory.")
    parser.add_argument("--class-map", required=True,
                        help="TSV with source_id and scannetpp_name columns.")
    parser.add_argument("--scene-id", default="custom_1file")
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--max-points-per-target-class", type=int, default=250_000,
                        help="Reservoir cap per ScanNet++ target class. 0 keeps all.")
    parser.add_argument("--max-total-points", type=int, default=0,
                        help="Optional final cap after class sampling. 0 disables.")
    parser.add_argument("--points-per-output-scene", type=int, default=250_000,
                        help="Split train/val into smaller .pth chunks for faster loading. 0 disables.")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--label-column", default="Classification",
                        help="Usually Classification for CloudCompare exports.")
    parser.add_argument("--instance-mode", choices=["none", "class"], default="none",
                        help="none is recommended when training with offset losses disabled.")
    return parser.parse_args()


def load_scannetpp_mapping(meta_dir):
    path = Path(meta_dir) / "label_mapping.tsv"
    name_to_id = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            name_to_id[row["source_label"].strip()] = int(row["class_index"])
    return name_to_id


def load_class_map(path, name_to_id):
    source_to_target = {}
    source_names = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            source_id = int(float(row["source_id"]))
            source_names[source_id] = row.get("source_name", "").strip()
            target_name = row.get("scannetpp_name", "").strip()
            if not target_name:
                continue
            if target_name not in name_to_id:
                raise ValueError(
                    "Unknown ScanNet++ class {!r} for source_id {}. "
                    "Check label_mapping.tsv.".format(target_name, source_id))
            source_to_target[source_id] = name_to_id[target_name]
    return source_to_target, source_names


def read_header_and_count(path):
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline().strip()
        if header.startswith("//"):
            header = header[2:]
        columns = header.split()
        count_line = f.readline().strip()
    try:
        declared_count = int(float(count_line))
    except ValueError:
        declared_count = None
    return columns, declared_count


def update_reservoir(existing, new_values, cap, rng):
    if new_values.size == 0:
        return existing
    if existing is None:
        if cap <= 0 or new_values.shape[0] <= cap:
            return new_values.copy()
        idx = rng.choice(new_values.shape[0], size=cap, replace=False)
        return new_values[idx].copy()
    merged = np.concatenate([existing, new_values], axis=0)
    if cap <= 0 or merged.shape[0] <= cap:
        return merged
    idx = rng.choice(merged.shape[0], size=cap, replace=False)
    return merged[idx].copy()


def copy_meta(base_meta, out_meta):
    out_meta.mkdir(parents=True, exist_ok=True)
    for name in [
        "classes.txt",
        "label_mapping.tsv",
        "panoptic_classes.tsv",
        "stuff_classes.txt",
        "thing_classes.txt",
    ]:
        src = Path(base_meta) / name
        if src.exists():
            shutil.copy2(src, out_meta / name)


def build_arrays(args, source_to_target, id_to_name):
    columns, declared_count = read_header_and_count(args.input_txt)
    required = ["X", "Y", "Z", "R", "G", "B", args.label_column]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("Missing required columns in {}: {}".format(args.input_txt, missing))

    usecols = [columns.index(name) for name in required]
    label_col_in_loaded = len(required) - 1
    rng = np.random.default_rng(args.seed)
    per_target = {}
    seen_source_counts = {}
    kept_target_counts = {}
    total_rows = 0

    print("Input columns:", " ".join(columns))
    if declared_count is not None:
        print("Declared points: {:,}".format(declared_count))
    print("Mapped source classes:", sorted(source_to_target.items()))

    with open(args.input_txt, "r", encoding="utf-8-sig", errors="replace") as f:
        f.readline()
        f.readline()
        while True:
            data = np.loadtxt(
                f,
                dtype=np.float32,
                max_rows=args.chunk_rows,
                usecols=usecols,
            )
            if data.size == 0:
                break
            if data.ndim == 1:
                data = data.reshape(1, -1)
            total_rows += data.shape[0]

            source_ids = np.rint(data[:, label_col_in_loaded]).astype(np.int64)
            unique_source, unique_counts = np.unique(source_ids, return_counts=True)
            for sid, cnt in zip(unique_source.tolist(), unique_counts.tolist()):
                seen_source_counts[int(sid)] = seen_source_counts.get(int(sid), 0) + int(cnt)

            for source_id, target_id in source_to_target.items():
                mask = source_ids == int(source_id)
                if not np.any(mask):
                    continue
                block = data[mask, :6].astype(np.float32, copy=False)
                target = np.full((block.shape[0], 1), int(target_id), dtype=np.float32)
                packed = np.concatenate([block, target], axis=1)
                per_target[target_id] = update_reservoir(
                    per_target.get(target_id),
                    packed,
                    int(args.max_points_per_target_class),
                    rng,
                )
                kept_target_counts[target_id] = int(per_target[target_id].shape[0])

            if total_rows % (args.chunk_rows * 5) == 0:
                print("Read {:,} rows, kept {} points".format(
                    total_rows, sum(kept_target_counts.values())), flush=True)

    if not per_target:
        raise RuntimeError("No mapped points found. Check --label-column and --class-map.")

    packed = np.concatenate([per_target[k] for k in sorted(per_target)], axis=0)
    if args.max_total_points and packed.shape[0] > args.max_total_points:
        idx = rng.choice(packed.shape[0], size=args.max_total_points, replace=False)
        packed = packed[idx]

    coord = packed[:, :3].astype(np.float32)
    rgb = packed[:, 3:6].astype(np.float32)
    sem = np.rint(packed[:, 6]).astype(np.int64)
    feat = rgb / 127.5 - 1.0

    if args.instance_mode == "class":
        inst = np.full(sem.shape, -1, dtype=np.int64)
        next_inst = 0
        for target_id in sorted(np.unique(sem).tolist()):
            target_name = id_to_name.get(int(target_id), "")
            # Instance ids are not needed for semantic finetuning. This mode is
            # only a coarse fallback for non-stuff labels.
            if target_name in STUFF_NAMES:
                continue
            mask = sem == int(target_id)
            inst[mask] = next_inst
            next_inst += 1
    else:
        inst = np.full(sem.shape, -1, dtype=np.int64)

    print("Rows read: {:,}".format(total_rows))
    print("Source class counts:", sorted(seen_source_counts.items()))
    print("Kept target counts:", sorted((int(k), int(v.shape[0])) for k, v in per_target.items()))
    print("Final points: {:,}".format(coord.shape[0]))
    return coord, feat.astype(np.float32), sem, inst


def save_chunk(path, coord, feat, sem, inst, idx):
    torch.save((
        coord[idx].astype(np.float32),
        feat[idx].astype(np.float32),
        sem[idx].astype(np.int64),
        inst[idx].astype(np.int64),
    ), path)
    print("Wrote {} with {:,} points".format(path, idx.shape[0]))


def write_split(output_root, scene_id, coord, feat, sem, inst, val_ratio, seed,
                points_per_output_scene):
    rng = np.random.default_rng(seed)
    n = coord.shape[0]
    idx = np.arange(n)
    rng.shuffle(idx)
    val_n = int(round(n * max(0.0, min(0.9, val_ratio))))
    if val_n <= 0 or val_n >= n:
        train_idx = idx
        val_idx = idx
    else:
        val_idx = idx[:val_n]
        train_idx = idx[val_n:]

    split_to_idx = {"train": train_idx, "val": val_idx}
    scene_split_rows = []
    for split, split_idx in split_to_idx.items():
        split_dir = output_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        if points_per_output_scene and split_idx.shape[0] > points_per_output_scene:
            chunks = [
                split_idx[i:i + points_per_output_scene]
                for i in range(0, split_idx.shape[0], points_per_output_scene)
            ]
        else:
            chunks = [split_idx]

        for chunk_id, chunk_idx in enumerate(chunks):
            chunk_scene_id = "{}_{:03d}".format(scene_id, chunk_id)
            out_path = split_dir / "{}.pth".format(chunk_scene_id)
            save_chunk(out_path, coord, feat, sem, inst, chunk_idx)
            scene_split_rows.append((chunk_scene_id, split))

    meta_dir = output_root / "meta"
    with open(meta_dir / "scene_split.tsv", "w", encoding="utf-8") as f:
        f.write("scene_id\tsplit\n")
        for chunk_scene_id, split in scene_split_rows:
            f.write("{}\t{}\n".format(chunk_scene_id, split))


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    copy_meta(args.base_meta, output_root / "meta")

    name_to_id = load_scannetpp_mapping(args.base_meta)
    id_to_name = {idx: name for name, idx in name_to_id.items()}
    source_to_target, _ = load_class_map(args.class_map, name_to_id)
    coord, feat, sem, inst = build_arrays(args, source_to_target, id_to_name)
    write_split(
        output_root, args.scene_id, coord, feat, sem, inst,
        args.val_ratio, args.seed, args.points_per_output_scene)
    print("Done:", output_root)


if __name__ == "__main__":
    main()
