from copy import deepcopy
import glob
import os
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.utils.data_util import data_prepare_scannet as data_prepare
from dataset.utils import transform_scannet as transform


seed = 0
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)


def TrainValCollateFn(batch):
    coord, feat, label = list(zip(*batch))
    offset, count = [], 0
    for item in coord:
        count += item.shape[0]
        offset.append(count)
    return {
        'coord': torch.cat(coord),
        'feat': torch.cat(feat),
        'target': torch.cat(label),
        'offset': torch.IntTensor(offset),
    }


def TestCollateFn(batch):
    coord, feat, label, pred_idx, offset = list(zip(*batch))
    return {
        'coord': torch.cat(coord),
        'feat': torch.cat(feat),
        'target': torch.cat(label),
        'offset': torch.IntTensor(np.cumsum(offset)),
        'pred_idx': torch.cat(pred_idx),
    }


class myImageFloder(Dataset):
    """ScanNet++ loader for preprocessed PointMixer .pth scenes.

    Expected directory:
      scannetpp_root/
        train/<scene_id>.pth
        val/<scene_id>.pth

    Each .pth is a tuple: (coord Nx3 float32, feat Nx3 float32, label N int64).
    Colors are expected in ScanNet style [-1, 1].
    """

    def __init__(self, args, mode, test_split=None):
        super().__init__()

        self.classes = int(args.classes)
        self.ignore_label = int(args.ignore_label)
        self.mode = mode
        self.data_root = deepcopy(args.scannetpp_root or args.scannet_semgseg_root)
        if self.data_root is None:
            raise ValueError('Set --scannetpp_root to the preprocessed ScanNet++ root.')
        self.voxel_size = float(args.voxel_size)

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
            'Run tools/prepare_scannetpp.py first.'
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

        print('ScanNet++ {}: {} scenes from {}'.format(
            self.mode, len(self.data_list), self.data_root))

    def __len__(self):
        return len(self.data_list) * self.loop

    def __getitem__(self, idx):
        data_idx = idx % len(self.data_list)
        data_path = self.data_list[data_idx]
        coord, feat, label = torch.load(data_path, map_location='cpu')

        coord = np.asarray(coord, dtype=np.float32)
        feat = np.asarray(feat, dtype=np.float32)
        label = np.asarray(label, dtype=np.int64)
        label[label < 0] = self.ignore_label

        coord, feat, label = data_prepare(
            coord, feat, label,
            self.mode, self.voxel_size, self.voxel_max,
            self.transform, self.shuffle_index)
        return coord, feat, label
