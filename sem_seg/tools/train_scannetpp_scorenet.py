#!/usr/bin/env python3
"""Train a proposal ScoreNet for PointMixer panoptic proposals."""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.infer_scannetpp_panoptic_scene import (
    build_cluster_proposals,
    build_model_args,
    load_pointmixer_panoptic,
    load_thing_classes,
    predict_scene,
)
from tools.infer_scannetpp_scene import read_label_map
from tools.scorenet_common import (
    ProposalScoreNet,
    proposal_feature_dim,
    proposal_to_feature,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scannetpp-root', required=True,
                        help='Prepared panoptic root, e.g. /datasets/pointmixer_scannetpp_top100_panoptic_fast008')
    parser.add_argument('--checkpoint', required=True,
                        help='Trained PointMixer panoptic checkpoint.')
    parser.add_argument('--label-map', required=True)
    parser.add_argument('--thing-classes', required=True)
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=2048)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0001)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--max-train-scenes', type=int, default=0,
                        help='0 means all train scenes.')
    parser.add_argument('--max-val-scenes', type=int, default=64,
                        help='0 means all val scenes.')
    parser.add_argument('--max-proposals-per-scene', type=int, default=384)
    parser.add_argument('--rebuild-cache', action='store_true')

    parser.add_argument('--classes', type=int, default=0)
    parser.add_argument('--max-points', type=int, default=16000)
    parser.add_argument('--min-points', type=int, default=1024)
    parser.add_argument('--block-size', type=float, default=2.0)
    parser.add_argument('--votes', type=int, default=1)
    parser.add_argument('--confidence-threshold', type=float, default=0.20)
    parser.add_argument('--ignore-label', type=int, default=255)
    parser.add_argument('--nsample', nargs='+', type=int, default=[8, 8, 8, 8, 8])
    parser.add_argument('--voxel-size', type=float, default=0.05)

    parser.add_argument('--proposal-radii', nargs='*', type=float, default=[0.10, 0.12, 0.15])
    parser.add_argument('--cluster-on', nargs='*', choices=['p', 'q'], default=['q'])
    parser.add_argument('--min-cluster-points', type=int, default=50)
    return parser.parse_args()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def scene_paths(root, split, limit):
    paths = sorted((Path(root) / split).glob('*.pth'))
    if limit and limit > 0:
        paths = paths[:limit]
    return paths


def build_gt_by_class(sem_label, inst_label, ignore_label):
    gt_by_class = {}
    valid_instances = np.unique(inst_label[inst_label >= 0])
    for inst_id in valid_instances.tolist():
        mask = inst_label == inst_id
        valid_sem = sem_label[mask]
        valid_sem = valid_sem[valid_sem != ignore_label]
        if valid_sem.size == 0:
            continue
        counts = np.bincount(valid_sem.astype(np.int64))
        class_idx = int(np.argmax(counts))
        indices = np.where(mask & (sem_label == class_idx))[0].astype(np.int64)
        if indices.size == 0:
            continue
        gt_by_class.setdefault(class_idx, []).append(np.sort(indices))
    return gt_by_class


def proposal_iou_with_gt(indices, gt_indices):
    inter = np.intersect1d(indices, gt_indices, assume_unique=True).shape[0]
    if inter == 0:
        return 0.0
    union = indices.shape[0] + gt_indices.shape[0] - inter
    return float(inter) / float(max(union, 1))


def proposal_target_iou(proposal, gt_by_class):
    candidates = gt_by_class.get(int(proposal['class_idx']), [])
    if not candidates:
        return 0.0
    indices = proposal['indices']
    best = 0.0
    for gt_indices in candidates:
        iou = proposal_iou_with_gt(indices, gt_indices)
        if iou > best:
            best = iou
    return best


def sample_proposals(proposals, targets, max_count):
    if not max_count or max_count <= 0 or len(proposals) <= max_count:
        return proposals, targets
    targets = np.asarray(targets, dtype=np.float32)
    positives = np.where(targets >= 0.25)[0]
    negatives = np.where(targets < 0.25)[0]
    keep_pos = positives[:max_count // 2]
    remaining = max_count - keep_pos.shape[0]
    if negatives.shape[0] > remaining:
        neg_scores = targets[negatives]
        hard_order = negatives[np.argsort(-neg_scores)]
        keep_neg = hard_order[:remaining]
    else:
        keep_neg = negatives
    keep = np.concatenate([keep_pos, keep_neg])
    if keep.shape[0] < max_count:
        extra_pool = np.setdiff1d(np.arange(len(proposals)), keep, assume_unique=False)
        if extra_pool.size:
            extra = extra_pool[:max_count - keep.shape[0]]
            keep = np.concatenate([keep, extra])
    keep = np.sort(keep)
    return [proposals[int(i)] for i in keep], targets[keep].tolist()


def collect_split(split, paths, args, model, classes, thing_classes):
    features = []
    targets = []
    proposal_count = 0
    for scene_idx, path in enumerate(paths, start=1):
        coord, feat, sem_label, inst_label = torch.load(str(path), map_location='cpu')
        xyz = np.asarray(coord, dtype=np.float32)
        feat = np.asarray(feat, dtype=np.float32)
        sem_label = np.asarray(sem_label, dtype=np.int64)
        inst_label = np.asarray(inst_label, dtype=np.int64)

        sem_pred, confidence, pt_offsets = predict_scene(
            model, xyz, feat, classes,
            args.max_points, args.min_points,
            args.block_size, args.votes,
        )
        proposals = build_cluster_proposals(
            xyz, sem_pred, confidence, pt_offsets, thing_classes,
            args.ignore_label, args.confidence_threshold,
            args.proposal_radii, args.min_cluster_points,
            args.cluster_on,
        )
        gt_by_class = build_gt_by_class(sem_label, inst_label, args.ignore_label)
        scene_targets = [proposal_target_iou(proposal, gt_by_class) for proposal in proposals]
        proposals, scene_targets = sample_proposals(
            proposals, scene_targets, args.max_proposals_per_scene)

        for proposal, target in zip(proposals, scene_targets):
            features.append(proposal_to_feature(
                xyz, sem_pred, confidence, pt_offsets,
                proposal, classes, xyz.shape[0]))
            targets.append(float(target))

        proposal_count += len(proposals)
        if scene_idx % 10 == 0 or scene_idx == len(paths):
            positives = int(np.sum(np.asarray(targets, dtype=np.float32) >= 0.25))
            print('[ScoreNet] {} scenes {}/{}: proposals={}, positives>=0.25={}'.format(
                split, scene_idx, len(paths), proposal_count, positives), flush=True)

    if features:
        return np.stack(features, axis=0).astype(np.float32), np.asarray(targets, dtype=np.float32)
    return np.zeros((0, proposal_feature_dim(classes)), dtype=np.float32), np.zeros((0,), dtype=np.float32)


def build_or_load_cache(args, model, classes, thing_classes):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / 'scorenet_proposals.pt'
    if cache_path.exists() and not args.rebuild_cache:
        print('[ScoreNet] loading proposal cache:', cache_path, flush=True)
        return torch.load(str(cache_path), map_location='cpu')

    train_paths = scene_paths(args.scannetpp_root, 'train', args.max_train_scenes)
    val_paths = scene_paths(args.scannetpp_root, 'val', args.max_val_scenes)
    print('[ScoreNet] train scenes:', len(train_paths), 'val scenes:', len(val_paths), flush=True)

    train_x, train_y = collect_split('train', train_paths, args, model, classes, thing_classes)
    val_x, val_y = collect_split('val', val_paths, args, model, classes, thing_classes)
    data = {
        'train_x': torch.from_numpy(train_x),
        'train_y': torch.from_numpy(train_y),
        'val_x': torch.from_numpy(val_x),
        'val_y': torch.from_numpy(val_y),
        'classes': int(classes),
        'feature_dim': int(proposal_feature_dim(classes)),
        'args': vars(args),
    }
    torch.save(data, str(cache_path))
    print('[ScoreNet] saved proposal cache:', cache_path, flush=True)
    return data


def train_scorenet(args, data):
    train_x, train_y = data['train_x'], data['train_y']
    val_x, val_y = data['val_x'], data['val_y']
    if train_x.shape[0] == 0:
        raise RuntimeError('No train proposals generated.')

    model = ProposalScoreNet(int(data['feature_dim'])).cuda()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_loader = DataLoader(
        TensorDataset(train_x, train_y),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        TensorDataset(val_x, val_y),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    ) if val_x.shape[0] else None

    output_dir = Path(args.output_dir)
    best_loss = float('inf')
    best_path = output_dir / 'scorenet_best.ckpt'
    last_path = output_dir / 'scorenet_last.ckpt'

    for epoch in range(args.epochs):
        model.train()
        train_losses = []
        for x, y in train_loader:
            x = x.cuda(non_blocking=True).float()
            y = y.cuda(non_blocking=True).float()
            logits = model(x)
            weights = 1.0 + 4.0 * y
            loss = (F.binary_cross_entropy_with_logits(logits, y, reduction='none') * weights).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = float('nan')
        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.cuda(non_blocking=True).float()
                    y = y.cuda(non_blocking=True).float()
                    logits = model(x)
                    loss = F.binary_cross_entropy_with_logits(logits, y)
                    val_losses.append(float(loss.detach().cpu()))
            val_loss = float(np.mean(val_losses)) if val_losses else float('nan')

        train_loss = float(np.mean(train_losses)) if train_losses else float('nan')
        print('[ScoreNet] epoch {}/{} train_loss={:.5f} val_loss={:.5f}'.format(
            epoch + 1, args.epochs, train_loss, val_loss), flush=True)

        payload = {
            'state_dict': model.state_dict(),
            'classes': int(data['classes']),
            'feature_dim': int(data['feature_dim']),
            'epoch': int(epoch),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'args': vars(args),
        }
        torch.save(payload, str(last_path))
        metric = val_loss if not np.isnan(val_loss) else train_loss
        if metric < best_loss:
            best_loss = metric
            torch.save(payload, str(best_path))
            print('[ScoreNet] saved best:', best_path, flush=True)

    return best_path


def main():
    args = parse_args()
    seed_all(args.seed)

    label_names = read_label_map(args.label_map)
    classes = args.classes or len(label_names)
    thing_classes = load_thing_classes(args.thing_classes, label_names, ['wall', 'floor', 'ceiling'], classes)

    model_args = build_model_args(args, classes)
    pointmixer = load_pointmixer_panoptic(args.checkpoint, model_args)
    data = build_or_load_cache(args, pointmixer, classes, thing_classes)
    print('[ScoreNet] train proposals:', int(data['train_x'].shape[0]),
          'val proposals:', int(data['val_x'].shape[0]), flush=True)
    print('[ScoreNet] positive train proposals >=0.25:',
          int((data['train_y'] >= 0.25).sum().item()), flush=True)
    best_path = train_scorenet(args, data)
    print('[ScoreNet] done. best checkpoint:', best_path, flush=True)


if __name__ == '__main__':
    main()
