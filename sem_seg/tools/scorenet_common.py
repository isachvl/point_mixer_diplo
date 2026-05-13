import math

import numpy as np
import torch
import torch.nn as nn


NUMERIC_FEATURE_DIM = 15


def proposal_feature_dim(classes):
    return int(classes) + NUMERIC_FEATURE_DIM


class ProposalScoreNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def score_cluster_heuristic(points, confidence):
    if points.shape[0] == 0:
        return 0.0
    mean_conf = float(confidence.mean())
    centroid = points.mean(axis=0)
    rms = float(np.sqrt(np.mean(np.sum((points - centroid) ** 2, axis=1))))
    compactness = 1.0 / (1.0 + rms)
    size_score = min(1.0, np.log1p(points.shape[0]) / np.log1p(2000.0))
    return mean_conf * (0.65 + 0.35 * size_score) * (0.65 + 0.35 * compactness)


def proposal_to_feature(xyz, sem_pred, confidence, pt_offsets, proposal, classes, scene_points):
    indices = proposal['indices']
    class_idx = int(proposal['class_idx'])
    shifted = xyz + pt_offsets
    pts = shifted[indices] if proposal.get('source') == 'q' else xyz[indices]
    conf = confidence[indices]
    offsets = pt_offsets[indices]

    centroid = pts.mean(axis=0)
    centered = pts - centroid
    rms = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1)))) if pts.shape[0] else 0.0
    bbox = np.ptp(pts, axis=0) if pts.shape[0] else np.zeros(3, dtype=np.float32)
    bbox_volume = float(np.prod(np.maximum(bbox, 1e-3)))
    offset_norm = np.linalg.norm(offsets, axis=1) if offsets.shape[0] else np.zeros(1, dtype=np.float32)
    radius = float(proposal.get('radius', 0.0))
    source_flag = 1.0 if proposal.get('source') == 'q' else 0.0
    n = int(indices.shape[0])

    numeric = np.asarray([
        math.log1p(n) / math.log1p(max(scene_points, 1)),
        float(n) / float(max(scene_points, 1)),
        float(conf.mean()) if conf.size else 0.0,
        float(conf.std()) if conf.size else 0.0,
        float(conf.min()) if conf.size else 0.0,
        float(conf.max()) if conf.size else 0.0,
        rms,
        1.0 / (1.0 + rms),
        float(bbox[0]),
        float(bbox[1]),
        float(bbox[2]),
        min(1.0, bbox_volume / 10.0),
        float(offset_norm.mean()) if offset_norm.size else 0.0,
        radius,
        source_flag,
    ], dtype=np.float32)

    one_hot = np.zeros((int(classes),), dtype=np.float32)
    if 0 <= class_idx < int(classes):
        one_hot[class_idx] = 1.0
    return np.concatenate([one_hot, numeric], axis=0).astype(np.float32)


def proposals_to_features(xyz, sem_pred, confidence, pt_offsets, proposals, classes):
    scene_points = int(xyz.shape[0])
    if not proposals:
        return np.zeros((0, proposal_feature_dim(classes)), dtype=np.float32)
    return np.stack([
        proposal_to_feature(xyz, sem_pred, confidence, pt_offsets, proposal, classes, scene_points)
        for proposal in proposals
    ], axis=0)


def load_scorenet_checkpoint(path, map_location='cpu'):
    checkpoint = torch.load(path, map_location=map_location)
    classes = int(checkpoint['classes'])
    input_dim = int(checkpoint.get('feature_dim', proposal_feature_dim(classes)))
    model = ProposalScoreNet(input_dim)
    model.load_state_dict(checkpoint['state_dict'])
    return model, checkpoint
