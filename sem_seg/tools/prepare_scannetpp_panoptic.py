#!/usr/bin/env python3
"""Convert ScanNet++ scenes to PointMixer panoptic training .pth files."""

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dataset.utils.voxelize import voxelize
from tools.prepare_scannetpp import (
    build_remap,
    collect_annotation_names,
    discover_scene_plys,
    labels_from_annotation,
    load_benchmark_mapping,
    load_segments_annotation,
    make_splits,
    read_coord_color_arrays,
    read_vertex_arrays,
    remap_labels,
    resolve_color_ply,
    scene_id_from_ply,
    update_label_name_votes,
    write_metadata,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True,
                        help='ScanNet++ data root, one scene dir, scans dir, or one semantic PLY.')
    parser.add_argument('--output-root', required=True,
                        help='Output root consumed by loader_scannetpp_panoptic.py.')
    parser.add_argument('--mesh-name', default='mesh_aligned_0.05_semantic.ply')
    parser.add_argument('--color-mesh-name', default='mesh_aligned_0.05.ply',
                        help='PLY in the same scans directory used for real RGB features. '
                             'Use an empty string to read RGB from --mesh-name.')
    parser.add_argument('--ignore-label', type=int, default=255)
    parser.add_argument('--ignore-raw-label', nargs='*', type=int, default=[-100])
    parser.add_argument('--label-source', choices=['benchmark', 'annotation', 'ply'], default='benchmark')
    parser.add_argument('--benchmark-dir', default=None)
    parser.add_argument('--benchmark-classes', default='top100.txt')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--val-scenes', nargs='*', default=None)
    parser.add_argument('--train-scenes', nargs='*', default=None)
    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--skip-label-names', action='store_true')
    parser.add_argument('--stuff-labels', nargs='*', default=['wall', 'floor', 'ceiling'],
                        help='Semantic labels trained without instance/offset targets.')
    parser.add_argument('--prevoxel-size', type=float, default=0.0,
                        help='Optional one-time scene downsample before saving. '
                             'Use 0.08-0.10 for faster training on HDD/low RAM.')
    return parser.parse_args()


def instances_from_annotation(ply_path, class_mapping, ignore_label, stuff_labels,
                              alias_to_class=None):
    segments, anno = load_segments_annotation(ply_path)
    seg_indices = np.asarray(segments.get('segIndices'), dtype=np.int64)
    if seg_indices.ndim != 1:
        raise ValueError('{} has invalid segIndices.'.format(ply_path.parent))

    stuff_labels = set(stuff_labels or [])
    seg_to_instance = {}
    next_instance = 0
    for group in anno.get('segGroups', []):
        name = str(group.get('label')).strip()
        if alias_to_class is not None:
            name = alias_to_class.get(name)
        if not name or name not in class_mapping or name in stuff_labels:
            continue

        instance_id = next_instance
        next_instance += 1
        for seg_id in group.get('segments', []):
            seg_to_instance[int(seg_id)] = instance_id

    instance = np.full(seg_indices.shape, -1, dtype=np.int64)
    if not seg_to_instance:
        return instance

    max_seg = int(seg_indices.max()) if seg_indices.size else -1
    if max_seg >= 0 and max_seg <= 20_000_000:
        lookup = np.full(max_seg + 1, -1, dtype=np.int64)
        for seg_id, instance_id in seg_to_instance.items():
            if 0 <= seg_id <= max_seg:
                lookup[seg_id] = instance_id
        valid = seg_indices >= 0
        instance[valid] = lookup[seg_indices[valid]]
    else:
        instance = np.asarray(
            [seg_to_instance.get(int(seg_id), -1) for seg_id in seg_indices],
            dtype=np.int64)
    return instance


def write_panoptic_metadata(output_root, mapping, stuff_labels):
    meta_dir = output_root / 'meta'
    label_to_idx = {str(raw): int(target) for raw, target in mapping.items()}
    stuff_ids = []
    thing_ids = []
    for raw, target in sorted(mapping.items(), key=lambda item: item[1]):
        if str(raw) in set(stuff_labels):
            stuff_ids.append(int(target))
        else:
            thing_ids.append(int(target))

    (meta_dir / 'stuff_classes.txt').write_text(
        '\n'.join(str(x) for x in stuff_ids) + ('\n' if stuff_ids else ''),
        encoding='utf-8')
    (meta_dir / 'thing_classes.txt').write_text(
        '\n'.join(str(x) for x in thing_ids) + ('\n' if thing_ids else ''),
        encoding='utf-8')
    with open(meta_dir / 'panoptic_classes.tsv', 'w', encoding='utf-8') as f:
        f.write('class_index\tclass_name\tis_thing\n')
        for raw, target in sorted(mapping.items(), key=lambda item: item[1]):
            is_thing = 0 if str(raw) in set(stuff_labels) else 1
            f.write('{}\t{}\t{}\n'.format(target, raw, is_thing))


def prevoxel_downsample(coord, feat, label, instance, voxel_size):
    if voxel_size is None or voxel_size <= 0:
        return coord, feat, label, instance

    shifted = coord.astype(np.float32).copy()
    shifted -= shifted.min(axis=0, keepdims=True)
    idx_sort, count = voxelize(shifted, voxel_size, mode=1)
    starts = np.cumsum(np.insert(count, 0, 0)[:-1])
    keep_idx = np.sort(idx_sort[starts]).astype(np.int64)
    return coord[keep_idx], feat[keep_idx], label[keep_idx], instance[keep_idx]


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    train_dir = output_root / 'train'
    val_dir = output_root / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    scene_plys = discover_scene_plys(args.input_root, args.mesh_name)
    if args.max_scenes is not None:
        scene_plys = scene_plys[:args.max_scenes]
    if not scene_plys:
        raise RuntimeError('No {} files found under {}'.format(args.mesh_name, args.input_root))

    scene_by_id = {scene_id_from_ply(path): path for path in scene_plys}
    print('Found {} ScanNet++ scene(s).'.format(len(scene_by_id)))

    all_source_labels = set()
    label_name_votes = defaultdict(Counter)
    benchmark_alias_to_class = None
    benchmark_mapping = None
    if args.label_source == 'benchmark':
        benchmark_mapping, benchmark_alias_to_class = load_benchmark_mapping(
            args.input_root, args.benchmark_dir, args.benchmark_classes)

    for scene_id, ply_path in scene_by_id.items():
        _, _, raw_label = read_vertex_arrays(ply_path)
        raw_unique = np.unique(raw_label)
        if args.label_source == 'benchmark':
            scene_labels = collect_annotation_names(ply_path, benchmark_alias_to_class)
            all_source_labels.update(scene_labels)
            label_desc = '{} benchmark labels'.format(len(scene_labels))
        elif args.label_source == 'annotation':
            scene_labels = collect_annotation_names(ply_path)
            all_source_labels.update(scene_labels)
            label_desc = '{} annotation labels'.format(len(scene_labels))
        else:
            all_source_labels.update(int(x) for x in raw_unique)
            label_desc = '{} raw labels'.format(len(raw_unique))
        if args.label_source == 'ply' and not args.skip_label_names:
            update_label_name_votes(ply_path, raw_label, label_name_votes)
        print('{}: {:,} vertices, {}'.format(scene_id, raw_label.shape[0], label_desc))

    if args.label_source == 'benchmark':
        mapping = {name: idx for name, idx in benchmark_mapping.items()
                   if name in all_source_labels}
        mapping = {name: idx for idx, name in enumerate(
            sorted(mapping, key=lambda key: benchmark_mapping[key]))}
    elif args.label_source == 'annotation':
        mapping = {name: idx for idx, name in enumerate(sorted(all_source_labels))}
    else:
        mapping = build_remap(all_source_labels, args.ignore_raw_label)
    if not mapping:
        raise RuntimeError('No trainable labels found.')

    train_ids, val_ids = make_splits(list(scene_by_id.keys()), args)
    unknown = set(train_ids + val_ids) - set(scene_by_id.keys())
    if unknown:
        raise RuntimeError('Unknown split scene ids: {}'.format(sorted(unknown)))

    split_dirs = {'train': train_dir, 'val': val_dir}
    splits = {'train': train_ids, 'val': val_ids}
    for split, scene_ids in splits.items():
        for scene_id in scene_ids:
            ply_path = scene_by_id[scene_id]
            _, _, raw_label = read_vertex_arrays(ply_path)
            color_ply_path = resolve_color_ply(ply_path, args.color_mesh_name)
            coord, color = read_coord_color_arrays(color_ply_path)
            if coord.shape[0] != raw_label.shape[0]:
                raise ValueError(
                    '{} RGB source vertex count ({}) does not match label source ({}).'.format(
                        scene_id, coord.shape[0], raw_label.shape[0]))
            feat = color / 127.5 - 1.0
            if args.label_source == 'benchmark':
                label = labels_from_annotation(
                    ply_path, mapping, args.ignore_label, benchmark_alias_to_class)
                instance = instances_from_annotation(
                    ply_path, mapping, args.ignore_label, args.stuff_labels,
                    benchmark_alias_to_class)
            elif args.label_source == 'annotation':
                label = labels_from_annotation(ply_path, mapping, args.ignore_label)
                instance = instances_from_annotation(
                    ply_path, mapping, args.ignore_label, args.stuff_labels)
            else:
                label = remap_labels(raw_label, mapping, args.ignore_label)
                instance = np.full(label.shape, -1, dtype=np.int64)

            if label.shape[0] != raw_label.shape[0] or instance.shape[0] != raw_label.shape[0]:
                raise ValueError('{} labels do not match PLY vertices.'.format(scene_id))
            instance[label == args.ignore_label] = -1

            before_count = label.shape[0]
            coord, feat, label, instance = prevoxel_downsample(
                coord, feat, label, instance, args.prevoxel_size)

            out_path = split_dirs[split] / '{}.pth'.format(scene_id)
            torch.save((
                coord.astype(np.float32),
                feat.astype(np.float32),
                label.astype(np.int64),
                instance.astype(np.int64),
            ), out_path)
            valid_sem = int(np.sum(label != args.ignore_label))
            valid_inst = int(np.sum(instance >= 0))
            num_inst = int(len(np.unique(instance[instance >= 0]))) if valid_inst else 0
            downsample_msg = ''
            if args.prevoxel_size > 0:
                downsample_msg = ', prevoxel {:.3f}: {:,}->{:,}'.format(
                    args.prevoxel_size, before_count, label.shape[0])
            print('Wrote {} ({}, {:,}/{:,} semantic, {:,} instance points, {} instances{})'.format(
                out_path, split, valid_sem, label.shape[0], valid_inst, num_inst, downsample_msg))

    write_metadata(output_root, mapping, splits, label_name_votes)
    write_panoptic_metadata(output_root, mapping, args.stuff_labels)
    print('Classes: {}'.format(len(mapping)))
    print('Stuff labels:', ', '.join(args.stuff_labels) if args.stuff_labels else 'none')
    if len(scene_by_id) == 1:
        print('Only one scene found; it was written to both train and val for smoke training.')


if __name__ == '__main__':
    main()
