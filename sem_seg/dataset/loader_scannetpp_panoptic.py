from copy import deepcopy
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.utils.voxelize import voxelize
from dataset.utils import transform_scannet as transform


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def TrainValCollateFn(batch):
    coord, feat, sem_label, inst_label, offset_label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return {
        'coord': torch.cat(coord),
        'feat': torch.cat(feat),
        'target': torch.cat(sem_label),
        'target_instance': torch.cat(inst_label),
        'target_offset': torch.cat(offset_label),
        'offset': torch.IntTensor(offset),
    }


def TestCollateFn(batch):
    return TrainValCollateFn(batch)


def build_offset_targets(coord, sem_label, inst_label, ignore_label):
    target_offset = np.zeros((coord.shape[0], 3), dtype=np.float32)
    valid = (sem_label != ignore_label) & (inst_label >= 0)
    if not np.any(valid):
        return target_offset

    for inst_id in np.unique(inst_label[valid]):
        mask = valid & (inst_label == inst_id)
        if not np.any(mask):
            continue
        centroid = coord[mask].mean(axis=0)
        target_offset[mask] = centroid - coord[mask]
    return target_offset


def _valid_label_indices(sem_label, ignore_label):
    return np.where(sem_label != ignore_label)[0]


def _choose_crop_center(sem_label, split, ignore_label, center_class=None):
    valid_idx = _valid_label_indices(sem_label, ignore_label)
    if valid_idx.size == 0:
        return sem_label.shape[0] // 2

    if 'train' not in split:
        return int(valid_idx[valid_idx.size // 2])

    if center_class is not None:
        class_idx = np.where((sem_label != ignore_label) & (sem_label == int(center_class)))[0]
        if class_idx.size > 0:
            return int(np.random.choice(class_idx))

    return int(np.random.choice(valid_idx))


def _crop_indices(coord, sem_label, center_idx, voxel_max, split, ignore_label,
                  block_size=0.0, min_points_in_block=1024):
    if not voxel_max or sem_label.shape[0] <= voxel_max:
        return None

    center_idx = int(np.clip(center_idx, 0, sem_label.shape[0] - 1))
    valid_idx = _valid_label_indices(sem_label, ignore_label)
    if valid_idx.size == 0:
        valid_idx = np.arange(sem_label.shape[0])

    if 'train' in split and block_size and float(block_size) > 0:
        center = coord[center_idx]
        half = float(block_size) / 2.0
        block_idx = np.where(
            (sem_label != ignore_label)
            & (np.abs(coord[:, 0] - center[0]) <= half)
            & (np.abs(coord[:, 1] - center[1]) <= half)
        )[0]

        if block_idx.size >= int(min_points_in_block):
            if block_idx.size >= voxel_max:
                return np.random.choice(block_idx, size=voxel_max, replace=False)
            # PointMixer has several stride-4 downsampling stages. Tiny crops can
            # leave a single bottleneck point and break BatchNorm, so top up a
            # valid local block with nearest valid points instead of returning it
            # as-is.
            xy = coord[valid_idx, :2]
            d2 = np.sum((xy - center[:2][None, :]) ** 2, axis=1)
            nearest = valid_idx[np.argsort(d2)]
            merged = np.unique(np.concatenate([block_idx, nearest]))
            if merged.size > voxel_max:
                nearest_set = set(int(x) for x in nearest[:voxel_max * 2].tolist())
                local = np.array([x for x in merged.tolist() if int(x) in nearest_set], dtype=np.int64)
                if local.size >= voxel_max:
                    return np.random.choice(local, size=voxel_max, replace=False)
                return nearest[:voxel_max]
            return merged

        # If the XY block is too empty, keep locality with nearest valid points.
        xy = coord[valid_idx, :2]
        d2 = np.sum((xy - center[:2][None, :]) ** 2, axis=1)
        take = min(max(int(min_points_in_block), int(voxel_max)), valid_idx.size)
        nearest = valid_idx[np.argsort(d2)[:take]]
        if nearest.size > voxel_max:
            return np.random.choice(nearest, size=voxel_max, replace=False)
        return nearest

    init_idx = center_idx if 'train' in split else sem_label.shape[0] // 2
    return np.argsort(np.sum(np.square(coord - coord[init_idx]), 1))[:voxel_max]


def data_prepare_panoptic(
        coord, feat, sem_label, inst_label,
        split='train', voxel_size=0.04, voxel_max=None,
        data_transform=None, shuffle_index=False, ignore_label=255,
        center_class=None, block_size=0.0, min_points_in_block=1024):
    if data_transform:
        coord, feat = data_transform(coord, feat)

    if voxel_size:
        coord_min = np.min(coord, 0)
        coord = coord - coord_min
        uniq_idx = voxelize(coord, voxel_size)
        coord = coord[uniq_idx]
        feat = feat[uniq_idx]
        sem_label = sem_label[uniq_idx]
        inst_label = inst_label[uniq_idx]

    crop_center = _choose_crop_center(
        sem_label, split, ignore_label, center_class=center_class)
    crop_idx = _crop_indices(
        coord, sem_label, crop_center, voxel_max, split, ignore_label,
        block_size=block_size, min_points_in_block=min_points_in_block)
    if crop_idx is not None:
        coord = coord[crop_idx]
        feat = feat[crop_idx]
        sem_label = sem_label[crop_idx]
        inst_label = inst_label[crop_idx]

    if shuffle_index:
        shuf_idx = np.arange(coord.shape[0])
        np.random.shuffle(shuf_idx)
        coord = coord[shuf_idx]
        feat = feat[shuf_idx]
        sem_label = sem_label[shuf_idx]
        inst_label = inst_label[shuf_idx]

    coord_min = np.min(coord, 0)
    coord = coord - coord_min
    offset_label = build_offset_targets(coord, sem_label, inst_label, ignore_label)

    return (
        torch.FloatTensor(coord),
        torch.FloatTensor(feat),
        torch.LongTensor(sem_label),
        torch.LongTensor(inst_label),
        torch.FloatTensor(offset_label),
    )


class myImageFloder(Dataset):
    """ScanNet++ panoptic loader.

    Expected .pth tuple:
      (coord Nx3 float32, feat Nx3 float32, semantic_label N int64,
       instance_label N int64)
    """

    def __init__(self, args, mode, test_split=None):
        super().__init__()

        self.classes = int(args.classes)
        self.ignore_label = int(args.ignore_label)
        self.mode = mode
        self.data_root = deepcopy(args.scannetpp_root or args.scannet_semgseg_root)
        if self.data_root is None:
            raise ValueError('Set --scannetpp_root to the preprocessed ScanNet++ panoptic root.')
        self.voxel_size = float(args.voxel_size)
        self.block_size = float(getattr(args, 'block_size', 0.0) or 0.0)
        self.min_points_in_block = int(getattr(args, 'min_points_in_block', 1024) or 1024)
        self.min_train_points = int(getattr(args, 'min_train_points', 0) or 0)
        self.class_balance_prob = float(getattr(args, 'class_balance_prob', 0.0) or 0.0)
        self._scene_class_ids = []
        self._global_class_weight = {}

        data_list = []
        if 'train' in mode:
            data_list += glob.glob(os.path.join(self.data_root, 'train', '*.pth'))
        if 'val' in mode:
            data_list += glob.glob(os.path.join(self.data_root, 'val', '*.pth'))
        if 'test' in mode:
            data_list += glob.glob(os.path.join(self.data_root, 'test', '*.pth'))
            raise NotImplementedError('ScanNet++ test mode is not wired yet.')
        data_list = sorted(data_list)
        assert len(data_list) > 0, (
            f'No .pth files found for mode={mode} under {self.data_root}. '
            'Run tools/prepare_scannetpp_panoptic.py first.'
        )

        if mode == 'train' or mode == 'trainval':
            self.voxel_max = int(args.train_voxel_max)
            self.transform = transform.Compose([
                transform.RandomRotate(along_z=True),
                transform.RandomScale(scale_low=0.9, scale_high=1.1),
                transform.RandomDropColor(color_augment=0.0),
            ])
            self.shuffle_index = True
            self.loop = int(args.loop)
            self.data_list = data_list
            self._filter_small_train_scenes()
        elif mode == 'val':
            self.voxel_max = int(args.eval_voxel_max)
            self.transform = None
            self.shuffle_index = False
            self.loop = 1
            self.test_split = test_split
            if self.test_split is None:
                self.data_list = data_list
            else:
                raise NotImplementedError('Split validation chunks are not implemented for ScanNet++.')
        else:
            raise ValueError('no such mode: {}'.format(mode))

        print('ScanNet++ panoptic {}: {} scenes from {}'.format(
            self.mode, len(self.data_list), self.data_root))
        if 'train' in self.mode and self.class_balance_prob > 0:
            self._bootstrap_class_balance()

    def _filter_small_train_scenes(self):
        if self.min_train_points <= 0 or 'train' not in self.mode:
            return

        kept = []
        skipped = []
        for data_path in self.data_list:
            loaded = torch.load(data_path, map_location='cpu')
            sem_label = np.asarray(loaded[2], dtype=np.int64)
            valid_count = int(np.sum(sem_label != self.ignore_label))
            if valid_count >= self.min_train_points:
                kept.append(data_path)
            else:
                skipped.append((os.path.basename(data_path), valid_count))

        self.data_list = kept
        if skipped:
            print('ScanNet++ panoptic skipped small train scenes (<{} labeled pts): {}'.format(
                self.min_train_points, len(skipped)))
            for name, count in skipped[:20]:
                print('  - {}: {}'.format(name, count))
            if len(skipped) > 20:
                print('  ... and {} more'.format(len(skipped) - 20))
        if not self.data_list:
            raise RuntimeError('No train scenes left after min_train_points filtering.')

    def _bootstrap_class_balance(self):
        hist = {}
        self._scene_class_ids = []
        for data_path in self.data_list:
            loaded = torch.load(data_path, map_location='cpu')
            sem_label = np.asarray(loaded[2], dtype=np.int64)
            valid = sem_label != self.ignore_label
            if not np.any(valid):
                self._scene_class_ids.append([])
                continue
            cls, counts = np.unique(sem_label[valid], return_counts=True)
            cls_ids = [int(c) for c in cls.tolist() if int(c) >= 0]
            self._scene_class_ids.append(cls_ids)
            for c, n in zip(cls.tolist(), counts.tolist()):
                c = int(c)
                if c >= 0:
                    hist[c] = hist.get(c, 0) + int(n)

        if hist:
            cls = sorted(hist)
            freq = np.array([max(hist[c], 1) for c in cls], dtype=np.float64)
            inv = 1.0 / np.sqrt(freq)
            inv = inv / max(inv.sum(), 1e-12)
            self._global_class_weight = {
                int(c): float(w) for c, w in zip(cls, inv.tolist())
            }
        print('ScanNet++ panoptic class-balanced crops: prob={} block_size={} min_points={}'.format(
            self.class_balance_prob, self.block_size, self.min_points_in_block))

    def __len__(self):
        return len(self.data_list) * self.loop

    def _sample_center_class(self, data_idx):
        if (
                'train' not in self.mode
                or self.class_balance_prob <= 0
                or np.random.random() >= self.class_balance_prob
                or not self._global_class_weight):
            return None

        if data_idx >= len(self._scene_class_ids):
            return None
        scene_classes = self._scene_class_ids[data_idx]
        if not scene_classes:
            return None

        weights = np.array(
            [self._global_class_weight.get(int(c), 0.0) for c in scene_classes],
            dtype=np.float64)
        if not np.isfinite(weights).all() or weights.sum() <= 0:
            return None
        weights = weights / weights.sum()
        return int(np.random.choice(np.asarray(scene_classes, dtype=np.int64), p=weights))

    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        loaded = torch.load(data_path, map_location='cpu')
        if len(loaded) != 4:
            raise ValueError(
                '{} is not panoptic data. Expected '
                '(coord, feat, semantic_label, instance_label).'.format(data_path))
        coord, feat, sem_label, inst_label = loaded

        coord = np.asarray(coord, dtype=np.float32)
        feat = np.asarray(feat, dtype=np.float32)
        sem_label = np.asarray(sem_label, dtype=np.int64)
        inst_label = np.asarray(inst_label, dtype=np.int64)
        sem_label[sem_label < 0] = self.ignore_label
        inst_label[sem_label == self.ignore_label] = -1
        center_class = self._sample_center_class(data_idx)

        return data_prepare_panoptic(
            coord, feat, sem_label, inst_label,
            self.mode, self.voxel_size, self.voxel_max,
            self.transform, self.shuffle_index, self.ignore_label,
            center_class=center_class, block_size=self.block_size,
            min_points_in_block=self.min_points_in_block)
