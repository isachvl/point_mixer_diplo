#!/usr/bin/env python3
"""Compute class-balanced semantic loss weights for ScanNet++ .pth data."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--classes', type=int, default=0)
    parser.add_argument('--ignore-label', type=int, default=255)
    parser.add_argument('--power', type=float, default=0.5,
                        help='0 disables balancing, 0.5 is sqrt inverse frequency, 1.0 is full inverse frequency.')
    parser.add_argument('--min-weight', type=float, default=0.25)
    parser.add_argument('--max-weight', type=float, default=5.0)
    parser.add_argument('--split', default='train')
    return parser.parse_args()


def read_label_names(data_root, classes):
    names = {idx: 'class_{}'.format(idx) for idx in range(classes)}
    path = data_root / 'meta' / 'label_mapping.tsv'
    if not path.exists():
        return names
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            idx = int(row['class_index'])
            names[idx] = row.get('source_label', '') or names.get(idx, 'class_{}'.format(idx))
    return names


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    classes = int(args.classes)
    if classes <= 0:
        classes = int((data_root / 'meta' / 'classes.txt').read_text(encoding='utf-8').strip())

    files = sorted((data_root / args.split).glob('*.pth'))
    if not files:
        raise RuntimeError('No .pth files found in {}'.format(data_root / args.split))

    counts = np.zeros(classes, dtype=np.int64)
    for path in files:
        loaded = torch.load(path, map_location='cpu')
        if len(loaded) < 3:
            raise ValueError('{} does not contain semantic labels.'.format(path))
        sem = np.asarray(loaded[2], dtype=np.int64)
        valid = (sem >= 0) & (sem != args.ignore_label) & (sem < classes)
        if np.any(valid):
            counts += np.bincount(sem[valid], minlength=classes)[:classes]

    present = counts > 0
    if not np.any(present):
        raise RuntimeError('No valid labels found.')

    freq = counts.astype(np.float64) / max(float(counts.sum()), 1.0)
    mean_freq = freq[present].mean()
    weights = np.ones(classes, dtype=np.float64)
    weights[present] = (mean_freq / np.maximum(freq[present], 1e-12)) ** float(args.power)
    weights[~present] = float(args.max_weight)
    weights = np.clip(weights, float(args.min_weight), float(args.max_weight))
    weights = weights / weights[present].mean()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, weights.astype(np.float32))

    csv_path = output.with_suffix('.csv')
    names = read_label_names(data_root, classes)
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['class_index', 'class_name', 'count', 'weight'])
        for idx in range(classes):
            writer.writerow([idx, names.get(idx, 'class_{}'.format(idx)),
                             int(counts[idx]), float(weights[idx])])

    print('[PM WEIGHTS] scenes:', len(files))
    print('[PM WEIGHTS] valid points:', int(counts.sum()))
    print('[PM WEIGHTS] present classes:', int(present.sum()), '/', classes)
    print('[PM WEIGHTS] weight min/max/mean:',
          float(weights.min()), float(weights.max()), float(weights[present].mean()))
    print('[PM WEIGHTS] wrote:', output)
    print('[PM WEIGHTS] wrote:', csv_path)


if __name__ == '__main__':
    main()
