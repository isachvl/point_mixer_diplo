#!/usr/bin/env python3
"""Convert ScanNet++ semantic PLY scenes to PointMixer training .pth files."""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch


PLY_TYPES = {
    'char': 'i1',
    'int8': 'i1',
    'uchar': 'u1',
    'uint8': 'u1',
    'short': '<i2',
    'int16': '<i2',
    'ushort': '<u2',
    'uint16': '<u2',
    'int': '<i4',
    'int32': '<i4',
    'uint': '<u4',
    'uint32': '<u4',
    'float': '<f4',
    'float32': '<f4',
    'double': '<f8',
    'float64': '<f8',
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-root', required=True,
                        help='ScanNet++ data root, one scene dir, scans dir, or one semantic PLY.')
    parser.add_argument('--output-root', required=True,
                        help='Output root consumed by loader_scannetpp.py.')
    parser.add_argument('--mesh-name', default='mesh_aligned_0.05_semantic.ply')
    parser.add_argument('--color-mesh-name', default='mesh_aligned_0.05.ply',
                        help='PLY in the same scans directory used for real RGB features. '
                             'Use an empty string to read RGB from --mesh-name.')
    parser.add_argument('--ignore-label', type=int, default=255)
    parser.add_argument('--ignore-raw-label', nargs='*', type=int, default=[-100])
    parser.add_argument('--label-source', choices=['benchmark', 'annotation', 'ply'], default='benchmark',
                        help='benchmark maps annotations to ScanNet++ semantic benchmark top classes; '
                             'annotation maps raw segments_anno.json label names to classes; '
                             'ply keeps raw PLY integer labels.')
    parser.add_argument('--benchmark-dir', default=None,
                        help='Directory with map_benchmark.csv and top100.txt. Auto-detected by default.')
    parser.add_argument('--benchmark-classes', default='top100.txt',
                        help='Benchmark class list file inside --benchmark-dir.')
    parser.add_argument('--val-ratio', type=float, default=0.2)
    parser.add_argument('--val-scenes', nargs='*', default=None,
                        help='Optional scene ids to place in val.')
    parser.add_argument('--train-scenes', nargs='*', default=None,
                        help='Optional scene ids to place in train.')
    parser.add_argument('--max-scenes', type=int, default=None)
    parser.add_argument('--skip-label-names', action='store_true')
    return parser.parse_args()


def read_ply_header(stream):
    lines = []
    while True:
        line = stream.readline()
        if not line:
            raise ValueError('PLY header ended before end_header.')
        line = line.decode('ascii').rstrip('\r\n')
        lines.append(line)
        if line == 'end_header':
            return lines


def parse_ply_header(lines):
    ply_format = None
    vertex_count = None
    vertex_props = []
    current_element = None

    for line in lines:
        parts = line.split()
        if not parts:
            continue
        if parts[0] == 'format':
            ply_format = parts[1]
        elif parts[0] == 'element':
            current_element = parts[1]
            if current_element == 'vertex':
                vertex_count = int(parts[2])
        elif parts[0] == 'property' and current_element == 'vertex':
            if parts[1] == 'list':
                raise ValueError('List properties in vertex elements are not supported.')
            vertex_props.append((parts[2], parts[1]))

    if ply_format != 'binary_little_endian':
        raise ValueError('Only binary_little_endian PLY is supported, got {}'.format(ply_format))
    if vertex_count is None:
        raise ValueError('PLY has no vertex element.')
    return vertex_count, vertex_props


def read_vertex_arrays(ply_path):
    with open(ply_path, 'rb') as f:
        header = read_ply_header(f)
        vertex_count, vertex_props = parse_ply_header(header)
        dtype = []
        for name, type_name in vertex_props:
            if type_name not in PLY_TYPES:
                raise ValueError('Unsupported PLY property type: {}'.format(type_name))
            dtype.append((name, PLY_TYPES[type_name]))
        vertices = np.fromfile(f, dtype=np.dtype(dtype), count=vertex_count)

    required = ['x', 'y', 'z', 'red', 'green', 'blue', 'label']
    missing = [name for name in required if name not in vertices.dtype.names]
    if missing:
        raise ValueError('{} misses required vertex fields: {}'.format(ply_path, missing))

    coord = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1).astype(np.float32)
    color = np.stack([vertices['red'], vertices['green'], vertices['blue']], axis=1).astype(np.float32)
    label = vertices['label'].astype(np.int64)
    return coord, color, label


def read_coord_color_arrays(ply_path):
    with open(ply_path, 'rb') as f:
        header = read_ply_header(f)
        vertex_count, vertex_props = parse_ply_header(header)
        dtype = []
        for name, type_name in vertex_props:
            if type_name not in PLY_TYPES:
                raise ValueError('Unsupported PLY property type: {}'.format(type_name))
            dtype.append((name, PLY_TYPES[type_name]))
        vertices = np.fromfile(f, dtype=np.dtype(dtype), count=vertex_count)

    required = ['x', 'y', 'z', 'red', 'green', 'blue']
    missing = [name for name in required if name not in vertices.dtype.names]
    if missing:
        raise ValueError('{} misses required vertex fields: {}'.format(ply_path, missing))

    coord = np.stack([vertices['x'], vertices['y'], vertices['z']], axis=1).astype(np.float32)
    color = np.stack([vertices['red'], vertices['green'], vertices['blue']], axis=1).astype(np.float32)
    return coord, color


def discover_scene_plys(input_root, mesh_name):
    root = Path(input_root)
    if root.is_file():
        return [root]
    if (root / mesh_name).exists():
        return [root / mesh_name]
    if (root / 'scans' / mesh_name).exists():
        return [root / 'scans' / mesh_name]
    return sorted(root.glob('*/scans/{}'.format(mesh_name)))


def resolve_color_ply(label_ply_path, color_mesh_name):
    if color_mesh_name is None or str(color_mesh_name).strip() == '':
        return label_ply_path
    color_path = label_ply_path.parent / color_mesh_name
    if not color_path.exists():
        raise FileNotFoundError(
            'RGB source {} not found for {}. Pass --color-mesh-name \"\" '
            'to use RGB from --mesh-name.'.format(color_path, label_ply_path))
    return color_path


def scene_id_from_ply(ply_path):
    if ply_path.parent.name == 'scans':
        return ply_path.parent.parent.name
    return ply_path.stem


def infer_dataset_root(input_root):
    root = Path(input_root)
    if root.is_file():
        scans_dir = root.parent
        scene_dir = scans_dir.parent if scans_dir.name == 'scans' else scans_dir
        return scene_dir.parent
    if root.name == 'scans':
        return root.parent.parent
    if (root / 'scans').exists():
        return root.parent
    if root.name == 'data':
        return root.parent
    return root


def load_benchmark_mapping(input_root, benchmark_dir, benchmark_classes):
    if benchmark_dir is None:
        benchmark_dir = infer_dataset_root(input_root) / 'metadata' / 'semantic_benchmark'
    else:
        benchmark_dir = Path(benchmark_dir)

    class_path = benchmark_dir / benchmark_classes
    map_path = benchmark_dir / 'map_benchmark.csv'
    if not class_path.exists() or not map_path.exists():
        raise FileNotFoundError(
            'Benchmark label source needs {} and {}'.format(class_path, map_path))

    classes = [line.strip() for line in class_path.read_text(encoding='utf-8').splitlines()
               if line.strip()]
    class_set = set(classes)
    class_mapping = {name: idx for idx, name in enumerate(classes)}

    alias_to_class = {name: name for name in classes}
    with open(map_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = (row.get('class') or '').strip()
            target = (row.get('semantic_map_to') or '').strip()
            if not source:
                continue
            if source in class_set:
                alias_to_class[source] = source
            elif target in class_set:
                alias_to_class[source] = target

    return class_mapping, alias_to_class


def make_splits(scene_ids, args):
    scene_ids = sorted(scene_ids)
    if args.train_scenes is not None or args.val_scenes is not None:
        train = sorted(args.train_scenes or [])
        val = sorted(args.val_scenes or [])
        if not train:
            train = [sid for sid in scene_ids if sid not in set(val)]
        if not val:
            val = [scene_ids[-1]]
        return train, val

    if len(scene_ids) == 1:
        return scene_ids, scene_ids

    val_count = max(1, int(round(len(scene_ids) * args.val_ratio)))
    val = scene_ids[-val_count:]
    train = scene_ids[:-val_count]
    if not train:
        train = scene_ids
    return train, val


def load_segments_annotation(ply_path):
    scans_dir = ply_path.parent
    seg_path = scans_dir / 'segments.json'
    anno_path = scans_dir / 'segments_anno.json'
    if not seg_path.exists() or not anno_path.exists():
        raise FileNotFoundError(
            'Annotation label source needs both {} and {}'.format(seg_path, anno_path))

    with open(seg_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno = json.load(f)
    return segments, anno


def collect_annotation_names(ply_path, alias_to_class=None):
    _, anno = load_segments_annotation(ply_path)
    names = set()
    for group in anno.get('segGroups', []):
        name = str(group.get('label')).strip()
        if not name:
            continue
        if alias_to_class is not None:
            name = alias_to_class.get(name)
            if name is None:
                continue
        names.add(name)
    return names


def labels_from_annotation(ply_path, class_mapping, ignore_label, alias_to_class=None):
    segments, anno = load_segments_annotation(ply_path)
    seg_indices = np.asarray(segments.get('segIndices'), dtype=np.int64)
    if seg_indices.ndim != 1:
        raise ValueError('{} has invalid segIndices.'.format(ply_path.parent))

    seg_to_class = {}
    for group in anno.get('segGroups', []):
        name = str(group.get('label')).strip()
        if alias_to_class is not None:
            name = alias_to_class.get(name)
        if not name or name not in class_mapping:
            continue
        class_idx = class_mapping[name]
        for seg_id in group.get('segments', []):
            seg_to_class[int(seg_id)] = class_idx

    label = np.full(seg_indices.shape, ignore_label, dtype=np.int64)
    if not seg_to_class:
        return label

    max_seg = int(seg_indices.max()) if seg_indices.size else -1
    if max_seg >= 0 and max_seg <= 20_000_000:
        lookup = np.full(max_seg + 1, ignore_label, dtype=np.int64)
        for seg_id, class_idx in seg_to_class.items():
            if 0 <= seg_id <= max_seg:
                lookup[seg_id] = class_idx
        valid = seg_indices >= 0
        label[valid] = lookup[seg_indices[valid]]
    else:
        label = np.asarray(
            [seg_to_class.get(int(seg_id), ignore_label) for seg_id in seg_indices],
            dtype=np.int64)
    return label


def build_remap(raw_labels, ignore_raw_labels, ignore_negative=True):
    ignore = set(ignore_raw_labels)
    valid = []
    for label in sorted(raw_labels):
        label = int(label)
        if label in ignore:
            continue
        if ignore_negative and label < 0:
            continue
        valid.append(label)
    return {raw: idx for idx, raw in enumerate(valid)}


def remap_labels(labels, mapping, ignore_label):
    out = np.full(labels.shape, ignore_label, dtype=np.int64)
    for raw, target in mapping.items():
        out[labels == raw] = target
    return out


def update_label_name_votes(ply_path, raw_labels, votes):
    scans_dir = ply_path.parent
    seg_path = scans_dir / 'segments.json'
    anno_path = scans_dir / 'segments_anno.json'
    if not seg_path.exists() or not anno_path.exists():
        return

    with open(seg_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno = json.load(f)

    seg_indices = segments.get('segIndices')
    if seg_indices is None or len(seg_indices) != raw_labels.shape[0]:
        return

    seg_to_name = {}
    for group in anno.get('segGroups', []):
        name = group.get('label')
        for seg_id in group.get('segments', []):
            seg_to_name[int(seg_id)] = name

    for raw, seg_id in zip(raw_labels, seg_indices):
        name = seg_to_name.get(int(seg_id))
        if name:
            votes[int(raw)][name] += 1


def write_metadata(output_root, mapping, splits, label_name_votes):
    meta_dir = output_root / 'meta'
    meta_dir.mkdir(parents=True, exist_ok=True)

    (meta_dir / 'classes.txt').write_text(str(len(mapping)) + '\n', encoding='utf-8')

    with open(meta_dir / 'label_mapping.tsv', 'w', encoding='utf-8') as f:
        f.write('class_index\tsource_label\n')
        for raw, target in sorted(mapping.items(), key=lambda item: item[1]):
            f.write('{}\t{}\n'.format(target, raw))

    if label_name_votes:
        with open(meta_dir / 'label_names.tsv', 'w', encoding='utf-8') as f:
            f.write('class_index\traw_label\tlabel_name\tvotes\n')
            for raw, target in sorted(mapping.items(), key=lambda item: item[1]):
                counter = label_name_votes.get(raw, Counter())
                if counter:
                    name, votes = counter.most_common(1)[0]
                else:
                    name, votes = '', 0
                f.write('{}\t{}\t{}\t{}\n'.format(target, raw, name, votes))

    with open(meta_dir / 'scene_split.tsv', 'w', encoding='utf-8') as f:
        f.write('scene_id\tsplit\n')
        for split, scene_ids in splits.items():
            for scene_id in scene_ids:
                f.write('{}\t{}\n'.format(scene_id, split))


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
                if label.shape[0] != raw_label.shape[0]:
                    raise ValueError('{} annotation labels do not match PLY vertices.'.format(scene_id))
            elif args.label_source == 'annotation':
                label = labels_from_annotation(ply_path, mapping, args.ignore_label)
                if label.shape[0] != raw_label.shape[0]:
                    raise ValueError('{} annotation labels do not match PLY vertices.'.format(scene_id))
            else:
                label = remap_labels(raw_label, mapping, args.ignore_label)
            out_path = split_dirs[split] / '{}.pth'.format(scene_id)
            torch.save((coord.astype(np.float32), feat.astype(np.float32), label), out_path)
            valid = int(np.sum(label != args.ignore_label))
            print('Wrote {} ({}, {:,}/{:,} labeled points)'.format(
                out_path, split, valid, label.shape[0]))

    write_metadata(output_root, mapping, splits, label_name_votes)
    print('Classes: {}'.format(len(mapping)))
    if len(scene_by_id) == 1:
        print('Only one scene found; it was written to both train and val for smoke training.')


if __name__ == '__main__':
    main()
