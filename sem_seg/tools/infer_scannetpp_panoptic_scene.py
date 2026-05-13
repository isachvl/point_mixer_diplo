#!/usr/bin/env python3
"""Run PointMixer panoptic inference on one ScanNet++/PLY scene."""

import argparse
import csv
import sys
from collections import Counter, defaultdict, deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model import get as get_model
from tools.infer_scannetpp_scene import (
    build_spatial_chunks,
    color_hex,
    color_rgb_text,
    downsample_input,
    infer_classes_from_checkpoint,
    make_palette,
    prepare_chunk,
    read_label_map,
    read_scene_ply,
    write_xlsx,
)
from tools.scorenet_common import (
    load_scorenet_checkpoint,
    proposals_to_features,
    score_cluster_heuristic,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scene-ply', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--label-map', required=True)
    parser.add_argument('--output-dir', required=True)

    parser.add_argument('--classes', type=int, default=0)
    parser.add_argument('--max-points', type=int, default=16000)
    parser.add_argument('--min-points', type=int, default=1024)
    parser.add_argument('--block-size', type=float, default=2.0)
    parser.add_argument('--votes', type=int, default=1)
    parser.add_argument('--confidence-threshold', type=float, default=0.0)
    parser.add_argument('--ignore-label', type=int, default=255)
    parser.add_argument('--write-point-csv', action='store_true')
    parser.add_argument('--top-k-log', type=int, default=0,
                        help='Write/print an explain log with top-K predicted classes and instances.')
    parser.add_argument('--explain-log', default=None,
                        help='Optional path for the top-K explain log. Defaults to output-dir/scene_topK_explain.txt.')

    parser.add_argument('--cluster-radius', type=float, default=0.12)
    parser.add_argument('--proposal-radii', nargs='*', type=float, default=None,
                        help='Optional multi-radius proposals for ScoreNet-lite/NMS. '
                             'Example: --proposal-radii 0.10 0.12 0.15')
    parser.add_argument('--cluster-on', nargs='*', choices=['p', 'q'], default=['q'],
                        help='Cluster on original coords p and/or shifted coords q.')
    parser.add_argument('--min-cluster-points', type=int, default=50)
    parser.add_argument('--cluster-score-threshold', type=float, default=0.20,
                        help='Drop weak candidate clusters before NMS.')
    parser.add_argument('--nms-iou-threshold', type=float, default=0.35,
                        help='Suppress lower-score clusters with point IoU above this threshold.')
    parser.add_argument('--scorenet-checkpoint', default=None,
                        help='Optional trained proposal ScoreNet checkpoint. '
                             'If omitted, ScoreNet-lite heuristic is used.')
    parser.add_argument('--thing-classes', default=None,
                        help='Optional txt file with one thing class index per line.')
    parser.add_argument('--stuff-class-names', nargs='*', default=['wall', 'floor', 'ceiling'])
    parser.add_argument('--confident-instances-only', action='store_true',
                        help='Write only clustered thing instances to confident outputs. '
                             'This hides stuff/background points such as wall/floor/ceiling.')
    parser.add_argument('--min-confident-instance-points', type=int, default=0,
                        help='Post-filter confident thing instances by final point count.')
    parser.add_argument('--min-confident-instance-score', type=float, default=None,
                        help='Post-filter confident thing instances by final instance score.')
    parser.add_argument('--exclude-class-names', nargs='*', default=[],
                        help='Class names to hide from confident outputs only.')
    parser.add_argument('--exclude-class-indices', nargs='*', type=int, default=[],
                        help='Class indices to hide from confident outputs only.')

    parser.add_argument('--nsample', nargs='+', type=int, default=[8, 8, 8, 8, 8])
    parser.add_argument('--voxel-size', type=float, default=0.05)
    parser.add_argument('--input-voxel-size', type=float, default=0.05,
                        help='Deterministic input downsample. Use 0.05 for dense pc_aligned.ply.')
    return parser.parse_args()


def build_model_args(args, classes):
    return SimpleNamespace(
        model='net_pointmixer_panoptic',
        arch='pointmixer_panoptic',
        intraLayer='PointMixerIntraSetLayer',
        interLayer='PointMixerInterSetLayer',
        transdown='SymmetricTransitionDownBlock',
        transup='SymmetricTransitionUpBlock',
        nsample=args.nsample,
        downsample=[1, 4, 4, 4, 4],
        drop_rate=0.1,
        fea_dim=6,
        classes=classes,
        ignore_label=args.ignore_label,
        print_freq=1000,
        optim='SGD',
        train_batch=1,
        val_batch=1,
        MYCHECKPOINT=str(args.output_dir),
        voxel_size=args.voxel_size,
        off_text_logger=True,
        on_train=False,
        offset_loss_weight=1.0,
        offset_dir_loss_weight=0.2,
    )


def load_thing_classes(path, label_names, stuff_names, classes):
    if path:
        values = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    values.append(int(line))
        return set(values)

    stuff_names = set(stuff_names or [])
    return {
        idx for idx in range(classes)
        if label_names.get(idx, 'class_{}'.format(idx)) not in stuff_names
    }


def resolve_class_filter(class_names, class_indices, label_names):
    selected = set(int(idx) for idx in (class_indices or []))
    if not class_names:
        return selected

    by_name = {}
    for idx, name in label_names.items():
        by_name.setdefault(str(name).strip().lower(), int(idx))

    missing = []
    for name in class_names:
        key = str(name).strip().lower()
        if not key:
            continue
        if key in by_name:
            selected.add(by_name[key])
        else:
            missing.append(name)
    if missing:
        print('[PM PANOPTIC] warning: unknown exclude class names:',
              ', '.join(missing), flush=True)
    return selected


def load_pointmixer_panoptic(checkpoint, model_args):
    ckpt = torch.load(checkpoint, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    state_dict = dict(state_dict)
    # Training checkpoints can store loss-only buffers that are not part of
    # the inference network. Dropping them keeps weight loading strict for the
    # actual model parameters.
    state_dict.pop('semantic_class_weight', None)
    model = get_model('net_pointmixer_panoptic')(args=model_args)
    model.load_state_dict(state_dict, strict=True)
    model.cuda()
    model.eval()
    return model.model


@torch.no_grad()
def predict_scene(model, xyz, feat, classes, max_points, min_points, block_size, votes):
    score_sum = np.zeros((xyz.shape[0], classes), dtype=np.float32)
    offset_sum = np.zeros((xyz.shape[0], 3), dtype=np.float32)
    vote_count = np.zeros((xyz.shape[0],), dtype=np.uint16)

    for vote in range(votes):
        chunks = build_spatial_chunks(xyz, block_size, max_points, vote, votes)
        print('[PM PANOPTIC] vote {}/{}: {} chunks'.format(vote + 1, votes, len(chunks)), flush=True)
        for chunk_id, idx in enumerate(chunks, start=1):
            coord, color, actual_n = prepare_chunk(xyz, feat, idx, min_points)
            coord_t = torch.from_numpy(coord).float().cuda(non_blocking=True).contiguous()
            feat_t = torch.from_numpy(color).float().cuda(non_blocking=True).contiguous()
            offset_t = torch.IntTensor([coord.shape[0]]).cuda(non_blocking=True)

            outputs = model([coord_t, feat_t, offset_t])
            logits = outputs['semantic_logits']
            pred_offsets = outputs['pt_offsets']
            probs = torch.softmax(logits[:actual_n], dim=1).detach().cpu().numpy()
            offsets = pred_offsets[:actual_n].detach().cpu().numpy()

            dst = idx[:actual_n]
            score_sum[dst] += probs
            offset_sum[dst] += offsets
            vote_count[dst] += 1

            if chunk_id % 25 == 0 or chunk_id == len(chunks):
                print('[PM PANOPTIC] chunk {}/{}'.format(chunk_id, len(chunks)), flush=True)

    covered = vote_count > 0
    score_sum[covered] /= vote_count[covered, None].astype(np.float32)
    offset_sum[covered] /= vote_count[covered, None].astype(np.float32)
    pred = np.argmax(score_sum, axis=1).astype(np.int32)
    conf = np.max(score_sum, axis=1).astype(np.float32)
    pred[~covered] = 255
    conf[~covered] = 0.0
    return pred, conf, offset_sum


def connected_components_radius(points, radius):
    if points.shape[0] == 0:
        return []

    radius2 = radius * radius
    cells = np.floor(points / radius).astype(np.int64)
    cell_map = defaultdict(list)
    for idx, cell in enumerate(cells):
        cell_map[tuple(cell.tolist())].append(idx)
    cell_map = {cell: np.asarray(items, dtype=np.int64) for cell, items in cell_map.items()}

    visited = np.zeros(points.shape[0], dtype=bool)
    components = []
    neighbor_delta = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
    ]

    for start in range(points.shape[0]):
        if visited[start]:
            continue
        visited[start] = True
        queue = deque([start])
        comp = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            cx, cy, cz = cells[cur]
            for dx, dy, dz in neighbor_delta:
                candidate_idx = cell_map.get((cx + dx, cy + dy, cz + dz))
                if candidate_idx is None:
                    continue
                unvisited = candidate_idx[~visited[candidate_idx]]
                if unvisited.size == 0:
                    continue
                diff = points[unvisited] - points[cur]
                near = unvisited[np.sum(diff * diff, axis=1) <= radius2]
                if near.size == 0:
                    continue
                visited[near] = True
                queue.extend(near.tolist())
        components.append(np.asarray(comp, dtype=np.int64))
    return components


def build_cluster_proposals(xyz, sem_pred, confidence, pt_offsets, thing_classes,
                            ignore_label, confidence_threshold, radii, min_points,
                            cluster_on):
    shifted = xyz + pt_offsets
    proposals = []
    coord_sources = {
        'p': xyz,
        'q': shifted,
    }

    for class_idx in sorted(thing_classes):
        if class_idx == ignore_label:
            continue
        mask = (sem_pred == class_idx) & (confidence >= confidence_threshold)
        global_idx = np.where(mask)[0].astype(np.int64)
        if global_idx.shape[0] < min_points:
            continue

        for source_name in cluster_on:
            coords = coord_sources[source_name][global_idx]
            for radius in radii:
                components = connected_components_radius(coords, radius)
                for comp in components:
                    if comp.shape[0] < min_points:
                        continue
                    pts = np.sort(global_idx[comp])
                    proposal_points = shifted[pts] if source_name == 'q' else xyz[pts]
                    proposals.append({
                        'class_idx': int(class_idx),
                        'indices': pts,
                        'score': score_cluster_heuristic(proposal_points, confidence[pts]),
                        'mean_confidence': float(confidence[pts].mean()),
                        'radius': float(radius),
                        'source': source_name,
                    })
    return proposals


def proposal_iou(a, b):
    inter = np.intersect1d(a, b, assume_unique=True).shape[0]
    if inter == 0:
        return 0.0
    union = a.shape[0] + b.shape[0] - inter
    return float(inter) / float(max(union, 1))


def nms_cluster_proposals(proposals, score_threshold, nms_iou_threshold):
    selected = []
    by_class = defaultdict(list)
    for proposal in proposals:
        if proposal['score'] >= score_threshold:
            by_class[proposal['class_idx']].append(proposal)

    for class_idx in sorted(by_class):
        candidates = sorted(by_class[class_idx], key=lambda item: item['score'], reverse=True)
        kept = []
        while candidates:
            best = candidates.pop(0)
            kept.append(best)
            survivors = []
            for candidate in candidates:
                if proposal_iou(best['indices'], candidate['indices']) <= nms_iou_threshold:
                    survivors.append(candidate)
            candidates = survivors
        selected.extend(kept)

    selected.sort(key=lambda item: item['score'], reverse=True)
    return selected


def score_proposals_with_model(proposals, xyz, sem_pred, confidence, pt_offsets,
                               classes, scorenet_checkpoint):
    if not scorenet_checkpoint or not proposals:
        return proposals
    model, checkpoint = load_scorenet_checkpoint(scorenet_checkpoint, map_location='cpu')
    model.cuda()
    model.eval()
    features = proposals_to_features(
        xyz, sem_pred, confidence, pt_offsets, proposals, int(checkpoint['classes']))
    scores = []
    with torch.no_grad():
        for start in range(0, features.shape[0], 4096):
            batch = torch.from_numpy(features[start:start + 4096]).float().cuda(non_blocking=True)
            pred = torch.sigmoid(model(batch)).detach().cpu().numpy()
            scores.extend(pred.tolist())
    for proposal, score in zip(proposals, scores):
        proposal['score'] = float(score)
        proposal['score_source'] = 'scorenet'
    return proposals


def cluster_instances(xyz, sem_pred, confidence, pt_offsets, thing_classes,
                      ignore_label, confidence_threshold, radius, min_points,
                      proposal_radii=None, cluster_on=None,
                      cluster_score_threshold=0.20, nms_iou_threshold=0.35,
                      scorenet_checkpoint=None, classes=None):
    instance_ids = np.full(sem_pred.shape, -1, dtype=np.int32)
    instance_scores = np.zeros(sem_pred.shape, dtype=np.float32)
    radii = proposal_radii if proposal_radii else [radius]
    cluster_on = cluster_on if cluster_on else ['q']

    proposals = build_cluster_proposals(
        xyz, sem_pred, confidence, pt_offsets, thing_classes,
        ignore_label, confidence_threshold, radii, min_points, cluster_on)
    if scorenet_checkpoint:
        proposals = score_proposals_with_model(
            proposals, xyz, sem_pred, confidence, pt_offsets, classes, scorenet_checkpoint)
    selected = nms_cluster_proposals(
        proposals, cluster_score_threshold, nms_iou_threshold)

    for instance_id, proposal in enumerate(selected):
        pts = proposal['indices']
        unclaimed = pts[instance_ids[pts] < 0]
        if unclaimed.shape[0] < min_points:
            continue
        instance_ids[unclaimed] = instance_id
        instance_scores[unclaimed] = float(proposal['score'])

    return instance_ids, instance_scores, proposals, selected


def make_panoptic_colors(sem_labels, instance_ids, class_palette):
    colors = class_palette[np.clip(sem_labels, 0, class_palette.shape[0] - 1)].copy()
    max_inst = int(instance_ids.max()) if np.any(instance_ids >= 0) else -1
    if max_inst >= 0:
        instance_palette = make_palette(max_inst + 1)
        for inst_id in range(max_inst + 1):
            colors[instance_ids == inst_id] = instance_palette[inst_id]
    return colors


def write_panoptic_ply(path, xyz, rgb, sem_labels, instance_ids, confidence, source_indices, instance_scores):
    dtype = np.dtype([
        ('x', '<f4'), ('y', '<f4'), ('z', '<f4'),
        ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
        ('class_label', '<i4'), ('instance_id', '<i4'),
        ('confidence', '<f4'), ('instance_score', '<f4'), ('source_index', '<i4'),
    ])
    out = np.empty(xyz.shape[0], dtype=dtype)
    out['x'], out['y'], out['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    out['red'], out['green'], out['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    out['class_label'] = sem_labels.astype(np.int32)
    out['instance_id'] = instance_ids.astype(np.int32)
    out['confidence'] = confidence.astype(np.float32)
    out['instance_score'] = instance_scores.astype(np.float32)
    out['source_index'] = source_indices.astype(np.int32)

    header = (
        'ply\n'
        'format binary_little_endian 1.0\n'
        'element vertex {}\n'
        'property float x\n'
        'property float y\n'
        'property float z\n'
        'property uchar red\n'
        'property uchar green\n'
        'property uchar blue\n'
        'property int class_label\n'
        'property int instance_id\n'
        'property float confidence\n'
        'property float instance_score\n'
        'property int source_index\n'
        'end_header\n'
    ).format(xyz.shape[0])
    with open(path, 'wb') as f:
        f.write(header.encode('ascii'))
        out.tofile(f)


def build_summary_rows(sem_labels, instance_ids, label_names, confidence, colors, kept_idx, instance_scores):
    rows = []
    used_pairs = sorted(set(zip(
        instance_ids[kept_idx].astype(np.int32).tolist(),
        sem_labels[kept_idx].astype(np.int32).tolist())))
    for inst_id, class_idx in used_pairs:
        mask = (instance_ids == inst_id) & (sem_labels == class_idx)
        mask_idx = np.where(mask & np.isin(np.arange(sem_labels.shape[0]), kept_idx))[0]
        if mask_idx.size == 0:
            continue
        rgb = colors[mask_idx[0]]
        rows.append([
            int(inst_id),
            int(class_idx),
            label_names.get(int(class_idx), 'class_{}'.format(int(class_idx))),
            int(mask_idx.size),
            float(confidence[mask_idx].mean()),
            float(instance_scores[mask_idx].mean()),
            color_rgb_text(rgb),
            color_hex(rgb),
            '',
        ])
    rows.sort(key=lambda row: row[3], reverse=True)
    return rows


def write_instance_summary_csv(path, rows):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'instance_id', 'class_index', 'class_name', 'count',
            'mean_confidence', 'mean_instance_score', 'color_rgb', 'color_hex',
        ])
        for row in rows:
            writer.writerow(row[:-1])


def write_instance_summary_xlsx(path, rows, label_names, class_palette):
    found_headers = [
        'instance_id',
        'class_index',
        'class_name',
        'count',
        'mean_confidence',
        'mean_instance_score',
        'color_rgb',
        'color_hex',
        'color_preview',
    ]
    legend_headers = [
        'class_index',
        'class_name',
        'color_rgb',
        'color_hex',
        'color_preview',
    ]
    legend_rows = []
    for idx in range(class_palette.shape[0]):
        rgb = class_palette[idx]
        legend_rows.append([
            int(idx),
            label_names.get(idx, 'class_{}'.format(idx)),
            color_rgb_text(rgb),
            color_hex(rgb),
            '',
        ])
    write_xlsx(path, found_headers, rows, legend_headers, legend_rows)


def write_point_csv(path, xyz, sem_labels, instance_ids, label_names, confidence,
                    colors, kept_idx, source_indices, instance_scores):
    with open(path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'point_index', 'source_point_index', 'x', 'y', 'z',
            'class_index', 'class_name', 'instance_id', 'confidence',
            'instance_score',
            'red', 'green', 'blue',
        ])
        for idx in kept_idx.tolist():
            class_idx = int(sem_labels[idx])
            writer.writerow([
                idx, int(source_indices[idx]),
                float(xyz[idx, 0]), float(xyz[idx, 1]), float(xyz[idx, 2]),
                class_idx,
                label_names.get(class_idx, 'class_{}'.format(class_idx)),
                int(instance_ids[idx]),
                float(confidence[idx]),
                float(instance_scores[idx]),
                int(colors[idx, 0]), int(colors[idx, 1]), int(colors[idx, 2]),
            ])


def build_top_class_rows(sem_labels, confidence, kept_idx, label_names, top_k):
    if top_k <= 0 or kept_idx.size == 0:
        return []

    kept_labels = sem_labels[kept_idx].astype(np.int32)
    kept_conf = confidence[kept_idx]
    total = max(int(kept_idx.size), 1)
    rows = []
    for rank, (class_idx, count) in enumerate(Counter(kept_labels.tolist()).most_common(top_k), start=1):
        mask = kept_labels == int(class_idx)
        class_conf = kept_conf[mask]
        rows.append({
            'rank': rank,
            'class_index': int(class_idx),
            'class_name': label_names.get(int(class_idx), 'class_{}'.format(int(class_idx))),
            'count': int(count),
            'share': float(count) / float(total),
            'mean_confidence': float(class_conf.mean()) if class_conf.size else 0.0,
            'max_confidence': float(class_conf.max()) if class_conf.size else 0.0,
        })
    return rows


def write_topk_explain_log(path, scene_name, args, xyz, all_idx, confident_idx,
                           sem_pred, confidence, label_names, all_rows, confident_rows,
                           proposals, selected_proposals):
    top_k = int(args.top_k_log)
    top_all_classes = build_top_class_rows(
        sem_pred, confidence, all_idx, label_names, top_k)
    top_confident_classes = build_top_class_rows(
        sem_pred, confidence, confident_idx, label_names, top_k)

    lines = []
    lines.append('PointMixer panoptic top-{} explain log'.format(top_k))
    lines.append('scene: {}'.format(scene_name))
    lines.append('points_after_input_voxel: {}'.format(int(xyz.shape[0])))
    lines.append('all_predicted_points: {}'.format(int(all_idx.shape[0])))
    lines.append('confident_clustered_points: {}'.format(int(confident_idx.shape[0])))
    lines.append('confidence_threshold: {:.4f}'.format(float(args.confidence_threshold)))
    lines.append('input_voxel_size: {:.4f}'.format(float(args.input_voxel_size)))
    lines.append('chunk_block_size: {:.4f}'.format(float(args.block_size)))
    lines.append('max_points_per_chunk: {}'.format(int(args.max_points)))
    lines.append('votes: {}'.format(int(args.votes)))
    lines.append('cluster_radius: {:.4f}'.format(float(args.cluster_radius)))
    lines.append('cluster_score_threshold: {:.4f}'.format(float(args.cluster_score_threshold)))
    lines.append('nms_iou_threshold: {:.4f}'.format(float(args.nms_iou_threshold)))
    lines.append('cluster_proposals_before_nms: {}'.format(len(proposals)))
    lines.append('cluster_proposals_selected_after_nms: {}'.format(len(selected_proposals)))
    lines.append('')
    lines.append('How semantic top classes are determined:')
    lines.append('- The model outputs logits for each point and class.')
    lines.append('- softmax(logits) gives class probabilities per point.')
    lines.append('- The predicted class is argmax(probabilities).')
    lines.append('- confidence is the max softmax probability for that point.')
    lines.append('- Top classes below are sorted by the number of predicted points.')
    lines.append('')
    lines.append('TOP CLASSES: all predicted semantic points')
    lines.append('rank\tclass_index\tclass_name\tpoints\tshare\tmean_conf\tmax_conf')
    for row in top_all_classes:
        lines.append('{rank}\t{class_index}\t{class_name}\t{count}\t{share:.4f}\t{mean_confidence:.4f}\t{max_confidence:.4f}'.format(**row))

    lines.append('')
    lines.append('TOP CLASSES: confident/clustered output')
    lines.append('rank\tclass_index\tclass_name\tpoints\tshare\tmean_conf\tmax_conf')
    for row in top_confident_classes:
        lines.append('{rank}\t{class_index}\t{class_name}\t{count}\t{share:.4f}\t{mean_confidence:.4f}\t{max_confidence:.4f}'.format(**row))

    lines.append('')
    lines.append('How instances are determined:')
    lines.append('- For thing classes, PointMixer also predicts a 3D offset vector per point.')
    lines.append('- The offset shifts points toward an estimated object center.')
    lines.append('- Shifted/original points are clustered by radius.')
    lines.append('- Candidate clusters are scored, then NMS removes overlapping duplicates.')
    lines.append('- Top instances below are sorted by final point count.')
    lines.append('')
    lines.append('TOP INSTANCES: all output')
    lines.append('rank\tinstance_id\tclass_index\tclass_name\tpoints\tmean_conf\tmean_instance_score')
    for rank, row in enumerate(all_rows[:top_k], start=1):
        lines.append('{}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}'.format(
            rank, int(row[0]), int(row[1]), row[2], int(row[3]),
            float(row[4]), float(row[5])))

    lines.append('')
    lines.append('TOP INSTANCES: confident output')
    lines.append('rank\tinstance_id\tclass_index\tclass_name\tpoints\tmean_conf\tmean_instance_score')
    for rank, row in enumerate(confident_rows[:top_k], start=1):
        lines.append('{}\t{}\t{}\t{}\t{}\t{:.4f}\t{:.4f}'.format(
            rank, int(row[0]), int(row[1]), row[2], int(row[3]),
            float(row[4]), float(row[5])))

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')

    print('[PM PANOPTIC] top-{} all classes:'.format(top_k), flush=True)
    for row in top_all_classes:
        print('[PM PANOPTIC] #{rank:02d} class {class_index}: {class_name} | points={count} share={share:.4f} mean_conf={mean_confidence:.4f}'.format(**row), flush=True)
    print('[PM PANOPTIC] wrote explain log:', path, flush=True)
    return path


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = output_dir

    label_names = read_label_map(args.label_map)
    classes = args.classes or len(label_names)
    ckpt = torch.load(args.checkpoint, map_location='cpu')
    state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt
    ckpt_classes = infer_classes_from_checkpoint(state_dict)
    if ckpt_classes:
        classes = ckpt_classes

    print('[PM PANOPTIC] reading scene:', args.scene_ply, flush=True)
    xyz, rgb = read_scene_ply(args.scene_ply)
    source_indices = np.arange(xyz.shape[0], dtype=np.int64)
    if args.input_voxel_size > 0:
        before = xyz.shape[0]
        xyz, rgb, source_indices = downsample_input(xyz, rgb, args.input_voxel_size)
        print('[PM PANOPTIC] input voxel downsample {:.4f}: {:,} -> {:,} points'.format(
            args.input_voxel_size, before, xyz.shape[0]), flush=True)

    feat = rgb.astype(np.float32) / 127.5 - 1.0
    thing_classes = load_thing_classes(
        args.thing_classes, label_names, args.stuff_class_names, classes)
    print('[PM PANOPTIC] points:', xyz.shape[0], 'classes:', classes,
          'thing classes:', len(thing_classes), flush=True)

    model_args = build_model_args(args, classes)
    model = load_pointmixer_panoptic(args.checkpoint, model_args)
    sem_pred, confidence, pt_offsets = predict_scene(
        model, xyz, feat, classes,
        args.max_points, args.min_points,
        args.block_size, args.votes,
    )
    instance_ids, instance_scores, proposals, selected_proposals = cluster_instances(
        xyz, sem_pred, confidence, pt_offsets, thing_classes,
        args.ignore_label, args.confidence_threshold,
        args.cluster_radius, args.min_cluster_points,
        args.proposal_radii, args.cluster_on,
        args.cluster_score_threshold, args.nms_iou_threshold,
        args.scorenet_checkpoint, classes,
    )

    class_palette = make_palette(classes)
    colors = make_panoptic_colors(sem_pred, instance_ids, class_palette)
    valid_semantic = sem_pred != args.ignore_label
    is_thing_pred = np.isin(sem_pred, np.asarray(sorted(thing_classes), dtype=np.int32))
    has_valid_instance = instance_ids >= 0
    all_idx = np.where(valid_semantic)[0].astype(np.int64)
    confident_mask = (
        valid_semantic
        & (confidence >= args.confidence_threshold)
        & (~is_thing_pred | has_valid_instance)
    )

    if args.confident_instances_only:
        confident_mask &= is_thing_pred & has_valid_instance

    if args.min_confident_instance_points > 0:
        keep_instance = np.zeros_like(has_valid_instance, dtype=bool)
        valid_ids = instance_ids[has_valid_instance].astype(np.int64)
        if valid_ids.size > 0:
            counts = np.bincount(valid_ids)
            keep_ids = np.flatnonzero(counts >= args.min_confident_instance_points)
            if keep_ids.size > 0:
                keep_instance = has_valid_instance & np.isin(instance_ids, keep_ids)
        confident_mask &= (~is_thing_pred | keep_instance)

    if args.min_confident_instance_score is not None:
        confident_mask &= (~is_thing_pred | (instance_scores >= args.min_confident_instance_score))

    exclude_classes = resolve_class_filter(
        args.exclude_class_names, args.exclude_class_indices, label_names)
    if exclude_classes:
        print('[PM PANOPTIC] hiding classes from confident outputs:',
              ', '.join([
                  '{}:{}'.format(idx, label_names.get(idx, 'class_{}'.format(idx)))
                  for idx in sorted(exclude_classes)
              ]), flush=True)
        confident_mask &= ~np.isin(
            sem_pred, np.asarray(sorted(exclude_classes), dtype=np.int32))

    confident_idx = np.where(confident_mask)[0].astype(np.int64)

    scene_name = Path(args.scene_ply).stem
    all_ply_path = output_dir / '{}_pointmixer_panoptic_all_colored.ply'.format(scene_name)
    confident_ply_path = output_dir / '{}_pointmixer_panoptic_confident_colored.ply'.format(scene_name)
    labels_npy = output_dir / '{}_pointmixer_panoptic_labels.npy'.format(scene_name)
    offsets_npy = output_dir / '{}_pointmixer_panoptic_offsets.npy'.format(scene_name)
    source_indices_npy = output_dir / '{}_pointmixer_source_indices.npy'.format(scene_name)
    all_summary_csv = output_dir / '{}_pointmixer_panoptic_all_instance_summary.csv'.format(scene_name)
    all_summary_xlsx = output_dir / '{}_pointmixer_panoptic_all_instance_summary.xlsx'.format(scene_name)
    confident_summary_csv = output_dir / '{}_pointmixer_panoptic_confident_instance_summary.csv'.format(scene_name)
    confident_summary_xlsx = output_dir / '{}_pointmixer_panoptic_confident_instance_summary.xlsx'.format(scene_name)

    np.save(labels_npy, np.stack([sem_pred, instance_ids], axis=1).astype(np.int32))
    np.save(offsets_npy, pt_offsets.astype(np.float32))
    np.save(source_indices_npy, source_indices)
    write_panoptic_ply(
        all_ply_path,
        xyz[all_idx],
        colors[all_idx],
        sem_pred[all_idx],
        instance_ids[all_idx],
        confidence[all_idx],
        source_indices[all_idx],
        instance_scores[all_idx],
    )
    write_panoptic_ply(
        confident_ply_path,
        xyz[confident_idx],
        colors[confident_idx],
        sem_pred[confident_idx],
        instance_ids[confident_idx],
        confidence[confident_idx],
        source_indices[confident_idx],
        instance_scores[confident_idx],
    )
    all_rows = build_summary_rows(
        sem_pred, instance_ids, label_names, confidence, colors, all_idx, instance_scores)
    confident_rows = build_summary_rows(
        sem_pred, instance_ids, label_names, confidence, colors, confident_idx, instance_scores)
    write_instance_summary_csv(all_summary_csv, all_rows)
    write_instance_summary_xlsx(all_summary_xlsx, all_rows, label_names, class_palette)
    write_instance_summary_csv(confident_summary_csv, confident_rows)
    write_instance_summary_xlsx(confident_summary_xlsx, confident_rows, label_names, class_palette)

    if args.top_k_log > 0:
        explain_log = args.explain_log
        if explain_log is None:
            explain_log = output_dir / '{}_pointmixer_panoptic_top{}_explain.txt'.format(
                scene_name, int(args.top_k_log))
        write_topk_explain_log(
            explain_log, scene_name, args, xyz, all_idx, confident_idx,
            sem_pred, confidence, label_names, all_rows, confident_rows,
            proposals, selected_proposals)

    if args.write_point_csv:
        all_point_csv = output_dir / '{}_pointmixer_panoptic_all_points.csv'.format(scene_name)
        confident_point_csv = output_dir / '{}_pointmixer_panoptic_confident_points.csv'.format(scene_name)
        write_point_csv(
            all_point_csv, xyz, sem_pred, instance_ids, label_names, confidence,
            colors, all_idx, source_indices, instance_scores)
        write_point_csv(
            confident_point_csv, xyz, sem_pred, instance_ids, label_names, confidence,
            colors, confident_idx, source_indices, instance_scores)
        print('[PM PANOPTIC] wrote:', all_point_csv, flush=True)
        print('[PM PANOPTIC] wrote:', confident_point_csv, flush=True)

    print('[PM PANOPTIC] all predicted points: {}/{}'.format(all_idx.shape[0], xyz.shape[0]), flush=True)
    print('[PM PANOPTIC] confident clustered points: {}/{}'.format(confident_idx.shape[0], xyz.shape[0]), flush=True)
    print('[PM PANOPTIC] cluster proposals:', len(proposals), 'selected after NMS:', len(selected_proposals), flush=True)
    print('[PM PANOPTIC] instances:', int(instance_ids.max() + 1) if np.any(instance_ids >= 0) else 0, flush=True)
    print('[PM PANOPTIC] wrote:', all_ply_path, flush=True)
    print('[PM PANOPTIC] wrote:', confident_ply_path, flush=True)
    print('[PM PANOPTIC] wrote:', labels_npy, flush=True)
    print('[PM PANOPTIC] wrote:', offsets_npy, flush=True)
    print('[PM PANOPTIC] wrote:', source_indices_npy, flush=True)
    print('[PM PANOPTIC] wrote:', all_summary_csv, flush=True)
    print('[PM PANOPTIC] wrote:', all_summary_xlsx, flush=True)
    print('[PM PANOPTIC] wrote:', confident_summary_csv, flush=True)
    print('[PM PANOPTIC] wrote:', confident_summary_xlsx, flush=True)
    print('[PM PANOPTIC] done', flush=True)


if __name__ == '__main__':
    main()
