#!/usr/bin/env python3
"""Compare one ScanNet++ scene annotation with saved PointMixer predictions."""

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-root', required=True,
                        help='ScanNet++ scene dir or scans dir, e.g. /datasets/scannetpp_full/data/00a231a370')
    parser.add_argument('--label-map', required=True,
                        help='TSV from pointmixer_scannetpp_top100/meta/label_mapping.tsv')
    parser.add_argument('--pred-labels', default=None,
                        help='Optional *_pointmixer_pred_labels.npy from infer_scannetpp_scene.py')
    parser.add_argument('--benchmark-dir', default=None,
                        help='Directory with ScanNet++ semantic_benchmark/map_benchmark.csv')
    parser.add_argument('--output-dir', default=None)
    parser.add_argument('--ignore-label', type=int, default=255)
    parser.add_argument('--top-k', type=int, default=5)
    return parser.parse_args()


def resolve_scans_dir(scene_root):
    root = Path(scene_root)
    if root.name == 'scans':
        return root
    if (root / 'scans').exists():
        return root / 'scans'
    return root


def infer_dataset_root(scene_root):
    root = Path(scene_root)
    if root.name == 'scans':
        return root.parent.parent.parent
    if (root / 'scans').exists():
        return root.parent.parent
    if root.name == 'data':
        return root.parent
    return root


def read_label_map(path):
    idx_to_name = {}
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            idx = int(row['class_index'])
            name = (row.get('source_label') or row.get('class_name') or '').strip()
            idx_to_name[idx] = name or 'class_{}'.format(idx)
    name_to_idx = {name: idx for idx, name in idx_to_name.items()}
    return idx_to_name, name_to_idx


def load_benchmark_alias(scene_root, benchmark_dir, idx_to_name):
    if benchmark_dir is None:
        benchmark_dir = infer_dataset_root(scene_root) / 'metadata' / 'semantic_benchmark'
    else:
        benchmark_dir = Path(benchmark_dir)

    map_path = benchmark_dir / 'map_benchmark.csv'
    if not map_path.exists():
        raise FileNotFoundError('Missing benchmark map: {}'.format(map_path))

    class_set = set(idx_to_name.values())
    alias_to_class = {name: name for name in class_set}
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
    return alias_to_class


def build_gt_labels(scans_dir, name_to_idx, alias_to_class, ignore_label):
    seg_path = scans_dir / 'segments.json'
    anno_path = scans_dir / 'segments_anno.json'
    if not seg_path.exists() or not anno_path.exists():
        raise FileNotFoundError('Need both {} and {}'.format(seg_path, anno_path))

    with open(seg_path, 'r', encoding='utf-8') as f:
        segments = json.load(f)
    with open(anno_path, 'r', encoding='utf-8') as f:
        anno = json.load(f)

    seg_indices = np.asarray(segments.get('segIndices'), dtype=np.int64)
    if seg_indices.ndim != 1:
        raise ValueError('{} has invalid segIndices'.format(seg_path))

    seg_to_class = {}
    for group in anno.get('segGroups', []):
        label_name = str(group.get('label') or '').strip()
        class_name = alias_to_class.get(label_name)
        if class_name not in name_to_idx:
            continue
        class_idx = int(name_to_idx[class_name])
        for seg_id in group.get('segments', []):
            seg_to_class[int(seg_id)] = class_idx

    gt = np.full(seg_indices.shape[0], ignore_label, dtype=np.int32)
    if not seg_to_class:
        return gt

    max_seg = max(max(seg_to_class), int(seg_indices.max(initial=0)))
    lookup = np.full(max_seg + 1, ignore_label, dtype=np.int32)
    for seg_id, class_idx in seg_to_class.items():
        if seg_id >= 0:
            lookup[seg_id] = class_idx

    valid = (seg_indices >= 0) & (seg_indices < lookup.shape[0])
    gt[valid] = lookup[seg_indices[valid]]
    return gt


def label_counts(labels, ignore_label):
    valid = labels != ignore_label
    return Counter(labels[valid].astype(np.int64).tolist())


def write_count_csv(path, counts, idx_to_name):
    rows = sorted(counts.items(), key=lambda item: item[1], reverse=True)
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_index', 'class_name', 'count'])
        for idx, count in rows:
            writer.writerow([int(idx), idx_to_name.get(int(idx), 'class_{}'.format(idx)), int(count)])


def write_metrics_csv(path, gt, pred, idx_to_name, ignore_label):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'class_index', 'class_name', 'gt_count', 'pred_count',
            'true_positive', 'false_positive', 'false_negative',
            'precision', 'recall', 'iou',
        ])
        for idx in sorted(idx_to_name):
            gt_mask = gt == idx
            pred_mask = pred == idx
            tp = int(np.count_nonzero(gt_mask & pred_mask))
            fp = int(np.count_nonzero((gt != idx) & pred_mask & (pred != ignore_label)))
            fn = int(np.count_nonzero(gt_mask & (pred != idx)))
            gt_count = int(np.count_nonzero(gt_mask))
            pred_count = int(np.count_nonzero(pred_mask))
            precision = tp / pred_count if pred_count else 0.0
            recall = tp / gt_count if gt_count else 0.0
            denom = tp + fp + fn
            iou = tp / denom if denom else 0.0
            writer.writerow([
                idx, idx_to_name[idx], gt_count, pred_count,
                tp, fp, fn, precision, recall, iou,
            ])


def write_gt_to_pred_csv(path, gt, pred, idx_to_name, ignore_label, top_k):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'gt_class_index', 'gt_class_name', 'gt_count',
            'pred_class_index', 'pred_class_name', 'pred_count', 'share_of_gt',
        ])
        for gt_idx in sorted(idx_to_name):
            mask = gt == gt_idx
            gt_count = int(np.count_nonzero(mask))
            if gt_count == 0:
                continue
            pred_counts = Counter(pred[mask & (pred != ignore_label)].astype(np.int64).tolist())
            for pred_idx, pred_count in pred_counts.most_common(top_k):
                writer.writerow([
                    gt_idx, idx_to_name[gt_idx], gt_count,
                    int(pred_idx), idx_to_name.get(int(pred_idx), 'class_{}'.format(pred_idx)),
                    int(pred_count), pred_count / gt_count,
                ])


def print_top(title, counts, idx_to_name, top_k):
    print('[PM CHECK] {}'.format(title), flush=True)
    for idx, count in counts.most_common(top_k):
        print('  {:>3} {:<28} {}'.format(
            int(idx), idx_to_name.get(int(idx), 'class_{}'.format(idx)), int(count)), flush=True)


def main():
    args = parse_args()
    scene_root = Path(args.scene_root)
    scans_dir = resolve_scans_dir(scene_root)
    output_dir = Path(args.output_dir) if args.output_dir else Path(args.pred_labels or scans_dir).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    idx_to_name, name_to_idx = read_label_map(args.label_map)
    alias_to_class = load_benchmark_alias(scene_root, args.benchmark_dir, idx_to_name)

    print('[PM CHECK] reading GT from:', scans_dir, flush=True)
    gt = build_gt_labels(scans_dir, name_to_idx, alias_to_class, args.ignore_label)
    gt_counts = label_counts(gt, args.ignore_label)

    scene_id = scans_dir.parent.name if scans_dir.name == 'scans' else scans_dir.name
    gt_summary = output_dir / '{}_gt_class_summary.csv'.format(scene_id)
    write_count_csv(gt_summary, gt_counts, idx_to_name)
    print_top('GT top classes:', gt_counts, idx_to_name, 10)
    print('[PM CHECK] wrote:', gt_summary, flush=True)

    if args.pred_labels:
        pred = np.load(args.pred_labels).astype(np.int32)
        if pred.shape[0] != gt.shape[0]:
            n = min(pred.shape[0], gt.shape[0])
            print('[PM CHECK] WARNING: pred points {} != GT points {}, comparing first {}'.format(
                pred.shape[0], gt.shape[0], n), flush=True)
            pred = pred[:n]
            gt = gt[:n]

        pred_counts = label_counts(pred, args.ignore_label)
        pred_summary = output_dir / '{}_pred_class_summary_from_npy.csv'.format(scene_id)
        metrics_csv = output_dir / '{}_pred_vs_gt_metrics.csv'.format(scene_id)
        confusion_csv = output_dir / '{}_gt_to_pred_top.csv'.format(scene_id)
        write_count_csv(pred_summary, pred_counts, idx_to_name)
        write_metrics_csv(metrics_csv, gt, pred, idx_to_name, args.ignore_label)
        write_gt_to_pred_csv(confusion_csv, gt, pred, idx_to_name, args.ignore_label, args.top_k)

        print_top('Prediction top classes:', pred_counts, idx_to_name, 10)
        wall_idx = name_to_idx.get('wall')
        chair_idx = name_to_idx.get('chair')
        if wall_idx is not None and chair_idx is not None:
            wall_mask = gt == wall_idx
            wall_count = int(np.count_nonzero(wall_mask))
            wall_as_chair = int(np.count_nonzero(wall_mask & (pred == chair_idx)))
            wall_as_wall = int(np.count_nonzero(wall_mask & (pred == wall_idx)))
            print('[PM CHECK] wall GT points:', wall_count, flush=True)
            print('[PM CHECK] wall predicted as wall:', wall_as_wall, flush=True)
            print('[PM CHECK] wall predicted as chair:', wall_as_chair, flush=True)

        print('[PM CHECK] wrote:', pred_summary, flush=True)
        print('[PM CHECK] wrote:', metrics_csv, flush=True)
        print('[PM CHECK] wrote:', confusion_csv, flush=True)


if __name__ == '__main__':
    main()
