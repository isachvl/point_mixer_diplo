#!/usr/bin/env python3
"""Evaluate PointMixer predictions against a custom annotation remapped to ScanNet++.

This script is for CloudCompare-style TXT annotations. It maps a custom
Classification column into ScanNet++ class ids, aligns annotation points to the
predicted/downsampled scene points, and reports semantic IoU/accuracy.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


PLY_TYPES = {
    "char": "i1", "uchar": "u1", "int8": "i1", "uint8": "u1",
    "short": "<i2", "ushort": "<u2", "int16": "<i2", "uint16": "<u2",
    "int": "<i4", "uint": "<u4", "int32": "<i4", "uint32": "<u4",
    "float": "<f4", "float32": "<f4", "double": "<f8", "float64": "<f8",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotation-txt", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--label-map", required=True)
    parser.add_argument("--pred-labels", required=True,
                        help="Panoptic labels .npy (Nx2) or semantic labels .npy (N).")
    parser.add_argument("--source-indices", required=True,
                        help="Inference *_source_indices.npy file.")
    parser.add_argument("--scene-ply", required=True,
                        help="Original scene PLY used for inference.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--match-by", choices=["xyz", "nearest-xyz", "source_index"], default="xyz")
    parser.add_argument("--label-column", default="Classification")
    parser.add_argument("--index-column", default="Original_cloud_index")
    parser.add_argument("--xyz-tolerance", type=float, default=1e-4,
                        help="Coordinate rounding tolerance for xyz matching.")
    parser.add_argument("--nearest-radius", type=float, default=0.05,
                        help="Radius for --match-by nearest-xyz.")
    parser.add_argument("--chunk-rows", type=int, default=1_000_000)
    parser.add_argument("--ignore-label", type=int, default=255)
    parser.add_argument("--classes", type=int, default=100)
    return parser.parse_args()


def read_ply_header(f):
    lines = []
    while True:
        line = f.readline()
        if not line:
            raise ValueError("Unexpected EOF while reading PLY header")
        text = line.decode("ascii", errors="replace").strip()
        lines.append(text)
        if text == "end_header":
            return lines


def parse_ply_header(lines):
    vertex_count = None
    props = []
    in_vertex = False
    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == "element":
            in_vertex = parts[1] == "vertex"
            if in_vertex:
                vertex_count = int(parts[2])
        elif in_vertex and parts[0] == "property" and len(parts) >= 3:
            props.append((parts[-1], parts[1]))
    if vertex_count is None:
        raise ValueError("PLY header has no vertex element")
    return vertex_count, props


def read_scene_xyz_by_indices(scene_ply, source_indices):
    with open(scene_ply, "rb") as f:
        header = read_ply_header(f)
        vertex_count, props = parse_ply_header(header)
        dtype = []
        for name, type_name in props:
            if type_name not in PLY_TYPES:
                raise ValueError("Unsupported PLY property type: {}".format(type_name))
            dtype.append((name, PLY_TYPES[type_name]))
        vertices = np.fromfile(f, dtype=np.dtype(dtype), count=vertex_count)

    for name in ["x", "y", "z"]:
        if name not in vertices.dtype.names:
            raise ValueError("{} misses {} property".format(scene_ply, name))
    if source_indices.max(initial=0) >= vertex_count:
        raise ValueError("source_indices exceed scene vertex count")
    return np.stack([
        vertices["x"][source_indices],
        vertices["y"][source_indices],
        vertices["z"][source_indices],
    ], axis=1).astype(np.float32)


def read_txt_columns(annotation_txt):
    with open(annotation_txt, "r", encoding="utf-8-sig", errors="replace") as f:
        header = f.readline().strip()
        if header.startswith("//"):
            header = header[2:]
        columns = header.split()
        count_line = f.readline().strip()
    try:
        count = int(float(count_line))
    except ValueError:
        count = None
    return columns, count


def read_label_names(label_map):
    names = {}
    with open(label_map, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            names[int(row["class_index"])] = row["source_label"]
    return names


def load_custom_to_scannet(class_map, label_names, ignore_label):
    name_to_id = {name: idx for idx, name in label_names.items()}
    max_source = 0
    rows = []
    with open(class_map, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            sid = int(float(row["source_id"]))
            max_source = max(max_source, sid)
            target_name = row.get("scannetpp_name", "").strip()
            if target_name:
                if target_name not in name_to_id:
                    raise ValueError("Unknown ScanNet++ class in map: {}".format(target_name))
                target = name_to_id[target_name]
            else:
                target = ignore_label
            rows.append((sid, target))

    lookup = np.full(max_source + 1, ignore_label, dtype=np.int64)
    for sid, target in rows:
        lookup[sid] = target
    return lookup


def remap_source_classes(source_cls, lookup, ignore_label):
    source_cls = np.rint(source_cls).astype(np.int64)
    out = np.full(source_cls.shape, ignore_label, dtype=np.int64)
    valid = (source_cls >= 0) & (source_cls < lookup.shape[0])
    out[valid] = lookup[source_cls[valid]]
    return out


def make_xyz_keys(xyz, tolerance):
    return np.rint(xyz.astype(np.float64) / float(tolerance)).astype(np.int64)


def structured_keys(keys):
    keys = np.ascontiguousarray(keys)
    return keys.view([("x", "<i8"), ("y", "<i8"), ("z", "<i8")]).reshape(-1)


def fill_gt_by_xyz(args, gt, lookup):
    pred_xyz = read_scene_xyz_by_indices(args.scene_ply, np.load(args.source_indices).astype(np.int64))
    target_keys = structured_keys(make_xyz_keys(pred_xyz, args.xyz_tolerance))
    order = np.argsort(target_keys)
    sorted_keys = target_keys[order]

    columns, declared_count = read_txt_columns(args.annotation_txt)
    required = ["X", "Y", "Z", args.label_column]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("Annotation TXT misses columns: {}".format(missing))
    usecols = [columns.index(name) for name in required]
    label_local_col = len(required) - 1

    matched = 0
    rows_read = 0
    with open(args.annotation_txt, "r", encoding="utf-8-sig", errors="replace") as f:
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
            rows_read += data.shape[0]

            ann_keys = structured_keys(make_xyz_keys(data[:, :3], args.xyz_tolerance))
            pos = np.searchsorted(sorted_keys, ann_keys)
            ok = (pos < sorted_keys.shape[0]) & (sorted_keys[pos] == ann_keys)
            if np.any(ok):
                pred_pos = order[pos[ok]]
                mapped = remap_source_classes(data[ok, label_local_col], lookup, args.ignore_label)
                gt[pred_pos] = mapped
                matched += int(ok.sum())

            if rows_read % (args.chunk_rows * 5) == 0:
                print("Read {:,} annotation rows, matched {:,}".format(rows_read, matched), flush=True)

    print("Annotation rows declared:", "{:,}".format(declared_count) if declared_count else "unknown")
    print("Annotation rows read: {:,}".format(rows_read))
    print("XYZ matched prediction points: {:,}/{}".format(np.sum(gt != -1), gt.shape[0]))
    return gt


def fill_gt_by_source_index(args, gt, lookup):
    source_indices = np.load(args.source_indices).astype(np.int64)
    order = np.argsort(source_indices)
    sorted_source = source_indices[order]
    columns, declared_count = read_txt_columns(args.annotation_txt)
    required = [args.index_column, args.label_column]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("Annotation TXT misses columns: {}".format(missing))
    usecols = [columns.index(name) for name in required]

    matched = 0
    rows_read = 0
    with open(args.annotation_txt, "r", encoding="utf-8-sig", errors="replace") as f:
        f.readline()
        f.readline()
        while True:
            data = np.loadtxt(
                f,
                dtype=np.float64,
                max_rows=args.chunk_rows,
                usecols=usecols,
            )
            if data.size == 0:
                break
            if data.ndim == 1:
                data = data.reshape(1, -1)
            rows_read += data.shape[0]

            ann_idx = np.rint(data[:, 0]).astype(np.int64)
            pos = np.searchsorted(sorted_source, ann_idx)
            ok = (pos < sorted_source.shape[0]) & (sorted_source[pos] == ann_idx)
            if np.any(ok):
                pred_pos = order[pos[ok]]
                mapped = remap_source_classes(data[ok, 1], lookup, args.ignore_label)
                gt[pred_pos] = mapped
                matched += int(ok.sum())

            if rows_read % (args.chunk_rows * 5) == 0:
                print("Read {:,} annotation rows, matched {:,}".format(rows_read, matched), flush=True)

    print("Annotation rows declared:", "{:,}".format(declared_count) if declared_count else "unknown")
    print("Annotation rows read: {:,}".format(rows_read))
    print("Source-index matched prediction points: {:,}/{}".format(np.sum(gt != -1), gt.shape[0]))
    return gt


def fill_gt_by_nearest_xyz(args, gt, lookup):
    try:
        from scipy.spatial import cKDTree
    except ImportError as exc:
        raise RuntimeError("--match-by nearest-xyz requires scipy in the runtime.") from exc

    pred_xyz = read_scene_xyz_by_indices(args.scene_ply, np.load(args.source_indices).astype(np.int64))
    tree = cKDTree(pred_xyz)
    best_dist = np.full(gt.shape, np.inf, dtype=np.float32)

    columns, declared_count = read_txt_columns(args.annotation_txt)
    required = ["X", "Y", "Z", args.label_column]
    missing = [name for name in required if name not in columns]
    if missing:
        raise ValueError("Annotation TXT misses columns: {}".format(missing))
    usecols = [columns.index(name) for name in required]
    label_local_col = len(required) - 1

    radius = float(args.nearest_radius)
    matched_updates = 0
    rows_read = 0
    with open(args.annotation_txt, "r", encoding="utf-8-sig", errors="replace") as f:
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
            rows_read += data.shape[0]

            try:
                dist, pos = tree.query(data[:, :3], k=1, distance_upper_bound=radius, workers=-1)
            except TypeError:
                dist, pos = tree.query(data[:, :3], k=1, distance_upper_bound=radius)
            ok = pos < gt.shape[0]
            if np.any(ok):
                cand_pos = pos[ok].astype(np.int64)
                cand_dist = dist[ok].astype(np.float32)
                cand_label = remap_source_classes(data[ok, label_local_col], lookup, args.ignore_label)

                # Keep only the nearest annotation for each prediction point in this chunk.
                order = np.lexsort((cand_dist, cand_pos))
                cand_pos = cand_pos[order]
                cand_dist = cand_dist[order]
                cand_label = cand_label[order]
                unique_pos, first_idx = np.unique(cand_pos, return_index=True)
                unique_dist = cand_dist[first_idx]
                unique_label = cand_label[first_idx]

                better = unique_dist < best_dist[unique_pos]
                if np.any(better):
                    update_pos = unique_pos[better]
                    best_dist[update_pos] = unique_dist[better]
                    gt[update_pos] = unique_label[better]
                    matched_updates += int(np.sum(better))

            if rows_read % (args.chunk_rows * 5) == 0:
                print("Read {:,} annotation rows, nearest-matched {:,}/{} prediction points".format(
                    rows_read, int(np.sum(gt != -1)), gt.shape[0]), flush=True)

    print("Annotation rows declared:", "{:,}".format(declared_count) if declared_count else "unknown")
    print("Annotation rows read: {:,}".format(rows_read))
    print("Nearest XYZ matched prediction points: {:,}/{} within radius {}".format(
        int(np.sum(gt != -1)), gt.shape[0], radius))
    print("Nearest match updates:", matched_updates)
    return gt


def compute_metrics(pred, gt, classes, ignore_label):
    valid = (gt >= 0) & (gt != ignore_label) & (gt < classes) & (pred >= 0) & (pred < classes)
    if not np.any(valid):
        raise RuntimeError("No valid GT points matched predictions. Try --match-by xyz or adjust --xyz-tolerance.")

    flat = gt[valid].astype(np.int64) * classes + pred[valid].astype(np.int64)
    conf = np.bincount(flat, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(conf)
    gt_count = conf.sum(axis=1)
    pred_count = conf.sum(axis=0)
    union = gt_count + pred_count - tp
    iou = np.full(classes, np.nan, dtype=np.float64)
    acc = np.full(classes, np.nan, dtype=np.float64)
    np.divide(tp, union, out=iou, where=union > 0)
    np.divide(tp, gt_count, out=acc, where=gt_count > 0)
    present = gt_count > 0
    return {
        "valid": valid,
        "confusion": conf,
        "tp": tp,
        "gt_count": gt_count,
        "pred_count": pred_count,
        "union": union,
        "iou": iou,
        "acc": acc,
        "present": present,
        "mIoU": float(np.nanmean(iou[present])),
        "mAcc": float(np.nanmean(acc[present])),
        "allAcc": float(tp.sum() / max(gt_count.sum(), 1)),
    }


def write_outputs(args, pred, gt, metrics, label_names):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "aligned_gt_scannetpp100_labels.npy", gt.astype(np.int16))
    np.save(out_dir / "confusion.npy", metrics["confusion"].astype(np.int64))

    with open(out_dir / "class_metrics.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_index", "class_name", "gt_count", "pred_count",
            "tp", "union", "iou", "acc",
        ])
        for idx in range(args.classes):
            if metrics["gt_count"][idx] <= 0 and metrics["pred_count"][idx] <= 0:
                continue
            writer.writerow([
                idx,
                label_names.get(idx, "class_{}".format(idx)),
                int(metrics["gt_count"][idx]),
                int(metrics["pred_count"][idx]),
                int(metrics["tp"][idx]),
                int(metrics["union"][idx]),
                "" if np.isnan(metrics["iou"][idx]) else "{:.6f}".format(metrics["iou"][idx]),
                "" if np.isnan(metrics["acc"][idx]) else "{:.6f}".format(metrics["acc"][idx]),
            ])

    lines = [
        "pred_points: {:,}".format(pred.shape[0]),
        "matched_gt_any_label: {:,}".format(int(np.sum(gt >= 0))),
        "valid_gt_mapped_points: {:,}".format(int(metrics["valid"].sum())),
        "present_gt_classes: {}".format(int(metrics["present"].sum())),
        "mIoU_gt_present: {:.6f}".format(metrics["mIoU"]),
        "mAcc_gt_present: {:.6f}".format(metrics["mAcc"]),
        "allAcc: {:.6f}".format(metrics["allAcc"]),
        "",
        "Top classes by GT count:",
    ]
    top = np.argsort(-metrics["gt_count"])[:30]
    for idx in top:
        if metrics["gt_count"][idx] <= 0:
            continue
        lines.append(
            "{:03d} {:<24} gt={:<10d} pred={:<10d} IoU={:.4f} Acc={:.4f}".format(
                idx,
                label_names.get(int(idx), "class_{}".format(int(idx)))[:24],
                int(metrics["gt_count"][idx]),
                int(metrics["pred_count"][idx]),
                0.0 if np.isnan(metrics["iou"][idx]) else float(metrics["iou"][idx]),
                0.0 if np.isnan(metrics["acc"][idx]) else float(metrics["acc"][idx]),
            )
        )
    (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)
    print("Wrote:", out_dir, flush=True)


def main():
    args = parse_args()
    pred_raw = np.load(args.pred_labels)
    pred = pred_raw[:, 0] if pred_raw.ndim == 2 else pred_raw
    pred = pred.astype(np.int64)
    gt = np.full(pred.shape, -1, dtype=np.int64)

    label_names = read_label_names(args.label_map)
    lookup = load_custom_to_scannet(args.class_map, label_names, args.ignore_label)
    if args.match_by == "xyz":
        gt = fill_gt_by_xyz(args, gt, lookup)
    elif args.match_by == "nearest-xyz":
        gt = fill_gt_by_nearest_xyz(args, gt, lookup)
    else:
        gt = fill_gt_by_source_index(args, gt, lookup)

    metrics = compute_metrics(pred, gt, args.classes, args.ignore_label)
    write_outputs(args, pred, gt, metrics, label_names)


if __name__ == "__main__":
    main()
