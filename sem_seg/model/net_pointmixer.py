from __future__ import print_function
import csv
import subprocess
import os
import pdb
import random
from datetime import datetime
curDT = datetime.now()
date_time = curDT.strftime("%Y-%m-%d %H:%M")
from copy import deepcopy
from collections import defaultdict

import cv2
import numpy as np
import open3d as o3d
import pytorch_lightning as pl
import matplotlib.pyplot as plt
from colorama import Fore, Back, Style

import torch
import torch.nn as nn
import torch.utils.data
# import torch.nn.functional as F
import torch.distributed as dist

from .network.get_network import get_network
from utils.common_util import AverageMeter, intersectionAndUnionGPU
from utils.logger import GOATLogger
from dataset import get as get_dataset

seed=0
pl.seed_everything(seed) # , workers=True
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # if use multi-GPU
# torch.backends.cudnn.deterministic=True
# torch.backends.cudnn.benchmark=False 

class net_pointmixer(pl.LightningModule):
    def __init__(self, args=None):
        super().__init__()
        
        # ------------
        # save_hyperparameters
        # ------------
        self.save_hyperparameters("args")
        args = self.hparams['args']        
        self.MYCHECKPOINT = deepcopy(args.MYCHECKPOINT)
        self.voxel_size = deepcopy(args.voxel_size)
        self.ignore_label = int(args.ignore_label) # 255
        self.classes = int(args.classes)
        self.class_names = self._load_class_names(args)
        self.print_freq = int(args.print_freq)
        self.optim = deepcopy(args.optim)

        self.train_batch = int(args.train_batch)
        self.val_batch = int(args.val_batch)

        # ------------
        # model
        # ------------        
        self.model = get_network(args)

        # ------------
        # metrics
        # ------------
        self.resetMetrics()

        # ------------
        # logger
        # ------------
        if not bool(args.off_text_logger): # FIXME. pytorch-lightning does not support text logger.
            mode = 'train' if bool(args.on_train) else 'test'
            self.text_logger = GOATLogger(
                mode=mode, 
                save_root=args.MYCHECKPOINT,
                log_freq=0,
                base_name=f'log_{mode}_{date_time}',
                n_iterations=0,
                n_eval_iterations=0)
        else:
            self.text_logger = None 
                
    def resetMetrics(self):
        self.intersection_meter = AverageMeter()
        self.union_meter = AverageMeter()
        self.target_meter = AverageMeter()
        self.nvox_meter = AverageMeter()

    def use_distributed_metrics(self):
        return dist.is_available() and dist.is_initialized()

    def _load_class_names(self, args):
        names = [str(i) for i in range(self.classes)]
        root = getattr(args, 'scannetpp_root', None)
        if not root:
            return names

        candidates = [
            os.path.join(root, 'meta', 'panoptic_classes.tsv'),
            os.path.join(root, 'meta', 'label_mapping.tsv'),
        ]
        for path in candidates:
            if not os.path.isfile(path):
                continue
            try:
                with open(path, 'r', encoding='utf-8', newline='') as f:
                    reader = csv.DictReader(f, delimiter='\t')
                    if not reader.fieldnames:
                        continue
                    idx_field = 'class_index' if 'class_index' in reader.fieldnames else reader.fieldnames[0]
                    if 'class_name' in reader.fieldnames:
                        name_field = 'class_name'
                    elif 'source_label' in reader.fieldnames:
                        name_field = 'source_label'
                    else:
                        name_field = reader.fieldnames[-1]
                    for row in reader:
                        try:
                            idx = int(row[idx_field])
                        except (TypeError, ValueError):
                            continue
                        if 0 <= idx < self.classes:
                            names[idx] = row.get(name_field, str(idx))
                return names
            except OSError as exc:
                if self.global_rank == 0:
                    print('[PM WARN] failed to read class names from {}: {}'.format(path, exc))
        return names

    def _save_val_class_metrics(self, iou_class, accuracy_class):
        metrics_dir = os.path.join(self.MYCHECKPOINT, 'class_metrics')
        os.makedirs(metrics_dir, exist_ok=True)

        rows = []
        for class_idx in range(self.classes):
            rows.append({
                'class_index': int(class_idx),
                'class_name': self.class_names[class_idx] if class_idx < len(self.class_names) else str(class_idx),
                'iou': float(iou_class[class_idx]),
                'accuracy': float(accuracy_class[class_idx]),
                'intersection': float(self.intersection_meter.sum[class_idx]),
                'union': float(self.union_meter.sum[class_idx]),
                'target': float(self.target_meter.sum[class_idx]),
            })
        rows = sorted(rows, key=lambda row: row['iou'], reverse=True)
        for rank, row in enumerate(rows, start=1):
            row['rank_by_iou'] = rank

        epoch = int(self.current_epoch)
        csv_path = os.path.join(metrics_dir, 'val_class_metrics_epoch_{:03d}.csv'.format(epoch))
        fieldnames = [
            'rank_by_iou', 'class_index', 'class_name', 'iou', 'accuracy',
            'intersection', 'union', 'target',
        ]
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        top20 = rows[:20]
        txt_path = os.path.join(metrics_dir, 'val_top20_epoch_{:03d}.txt'.format(epoch))
        top_lines = []
        for row in top20:
            top_lines.append(
                '{rank:02d}. id={idx:03d} {name} | IoU={iou:.4f} Acc={acc:.4f} target={target:.0f}'.format(
                    rank=row['rank_by_iou'],
                    idx=row['class_index'],
                    name=row['class_name'],
                    iou=row['iou'],
                    acc=row['accuracy'],
                    target=row['target'],
                ))
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(top_lines) + '\n')

        history_path = os.path.join(metrics_dir, 'val_top20_history.csv')
        write_header = not os.path.isfile(history_path)
        with open(history_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['epoch'] + fieldnames)
            if write_header:
                writer.writeheader()
            for row in top20:
                out = {'epoch': epoch}
                out.update(row)
                writer.writerow(out)

        msg = '[PM CLASS TOP20] epoch {} saved: {}\n{}'.format(
            epoch, csv_path, '\n'.join(top_lines))
        print(msg)
        if self.text_logger is not None:
            self.text_logger.loginfo(msg)

    def _loss_to_float(self, loss_value):
        if torch.is_tensor(loss_value):
            return float(loss_value.detach().mean().cpu().item())
        return float(loss_value)

    def _loss_scalars(self, losses, total_loss=None):
        scalars = {name: self._loss_to_float(value) for name, value in losses.items()}
        if total_loss is None:
            scalars['total_loss'] = sum(scalars.values())
        else:
            scalars['total_loss'] = self._loss_to_float(total_loss)
        return scalars

    def _format_loss_scalars(self, scalars):
        ordered_names = ['total_loss'] + sorted(
            name for name in scalars.keys() if name != 'total_loss')
        return ' '.join('{}[{:.6f}]'.format(name, scalars[name]) for name in ordered_names)

    def _append_loss_history(self, split, epoch, global_step, lr, nvox, scalars):
        metrics_dir = os.path.join(self.MYCHECKPOINT, 'loss_metrics')
        os.makedirs(metrics_dir, exist_ok=True)
        csv_path = os.path.join(metrics_dir, '{}_loss_history.csv'.format(split))
        fieldnames = ['epoch', 'global_step', 'lr', 'nvox'] + ['total_loss'] + sorted(
            name for name in scalars.keys() if name != 'total_loss')
        row = {
            'epoch': int(epoch),
            'global_step': int(global_step),
            'lr': float(lr),
            'nvox': int(nvox) if nvox is not None else '',
        }
        row.update(scalars)
        write_header = not os.path.isfile(csv_path)
        with open(csv_path, 'a', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if write_header:
                writer.writeheader()
            writer.writerow(row)

    def _aggregate_loss_outputs(self, outputs):
        if not outputs:
            return {}
        values = defaultdict(list)
        for output in outputs:
            if not output:
                continue
            for name, value in output.items():
                if name.startswith('loss/'):
                    values[name[5:]].append(self._loss_to_float(value))
        return {name: float(np.mean(vals)) for name, vals in values.items() if vals}

    def forward(self, data_dict):
        pred_dict = {}
        loss_dict = {}

        with torch.no_grad():
            coord = data_dict['coord']
            feat = data_dict['feat']
            target = data_dict['target']
            offset = data_dict['offset']

        output = self.model([coord, feat, offset])
        pred_dict['output'] = output
        if target.shape[-1] == 1:
            target = target[:, 0]  # for cls
        loss = nn.CrossEntropyLoss(ignore_index=self.ignore_label)(output, target)
        loss_dict['loss'] = loss

        return pred_dict, loss_dict
    
    @torch.no_grad()
    def calcMetrics(self, batch_idx, outputs, data_dict, logName='train/accuracy'):
        output = outputs['output'].detach()
        coord = data_dict['coord']
        target = data_dict['target']

        output = output.max(1)[1]
        n = coord.size(0)        
        count = target.new_tensor([n], dtype=torch.long)
        n = count.item()
        intersection, union, target = \
            intersectionAndUnionGPU(output, target, self.classes, self.ignore_label)
        # sync tensors in different gpus.
        if self.use_distributed_metrics():
            dist.all_reduce(intersection)
            dist.all_reduce(union)
            dist.all_reduce(target)
        intersection = intersection.cpu().detach().numpy()
        union = union.cpu().detach().numpy()
        target = target.cpu().detach().numpy()

        self.intersection_meter.update(intersection)
        self.union_meter.update(union)
        self.target_meter.update(target)

        accuracy = sum(self.intersection_meter.val) / (sum(self.target_meter.val) + 1e-10)
        if batch_idx % self.print_freq == 0:
            self.log(
                logName, accuracy, batch_size=1,
                sync_dist=self.use_distributed_metrics())

    def training_step(self, data_dict, batch_idx):
        outputs, losses = self.forward(data_dict)
        loss = sum(losses.values())

        if (self.global_rank == 0) and (self.global_step % 1000 == 0):

            nvox = float(data_dict['coord'].size(0)) / float(self.train_batch)
            loss_scalars = self._loss_scalars(losses, total_loss=loss)
            lr_value = float(self.scheduler.get_last_lr()[0])
            
            if self.logger is not None:
                self.logger.experiment["nvox_train"].log(nvox)
                self.logger.experiment["current_epoch_train"].log(self.current_epoch)
                self.logger.experiment["global_step_train"].log(self.global_step)
                self.logger.experiment["lr_train"].log(self.scheduler.get_last_lr())
                self.logger.experiment["loss_train"].log(loss_scalars['total_loss'])
                for loss_name, loss_value in loss_scalars.items():
                    self.logger.experiment["loss_train_{}".format(loss_name)].log(loss_value)
            if self.text_logger is not None:
                str_to_print = (
                    f'train : epoch[{self.current_epoch:d}], steps[{self.global_step:d}] lr[{lr_value:.6f}] | '
                    f'{self._format_loss_scalars(loss_scalars)}, '
                    f'nvox[{int(nvox):d}], '
                )
                self.text_logger.loginfo(str_to_print)
            self._append_loss_history(
                'train',
                self.current_epoch,
                self.global_step,
                lr_value,
                nvox,
                loss_scalars)
            print("TRAIN: epoch[%d], global_step[%d]: \n"%(self.current_epoch, self.global_step))
    
        self.resetMetrics() ######### WRONG
        return loss

    @torch.no_grad()
    def training_epoch_end(self, outputs):
        # train_step -> val_step -> val_epoch_end -> train_epoch_end 
        self.scheduler.step()

    @torch.no_grad()
    def validation_step(self, data_dict, batch_idx):
        with torch.no_grad():
            outputs, losses = self.forward(data_dict)
            self.calcMetrics(batch_idx, outputs, data_dict, logName='val/accuracy')
            loss = sum(losses.values())
            loss_scalars = self._loss_scalars(losses, total_loss=loss)
            return {'loss/{}'.format(name): value for name, value in loss_scalars.items()}
         
    @torch.no_grad()
    def validation_epoch_end(self, outputs):
        with torch.no_grad():
            val_loss_scalars = self._aggregate_loss_outputs(outputs)
            iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
            accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
            
            self.log(
                "mIoU_val", mIoU, batch_size=1,
                sync_dist=self.use_distributed_metrics())
            if self.global_rank == 0:
                if self.logger is not None:
                    self.logger.experiment["mIoU_val"].log(mIoU)
                    self.logger.experiment["mAcc_val"].log(mAcc)
                    self.logger.experiment["allAcc_val"].log(allAcc)
                    if 'total_loss' in val_loss_scalars:
                        self.logger.experiment["loss_val"].log(val_loss_scalars['total_loss'])
                    for loss_name, loss_value in val_loss_scalars.items():
                        self.logger.experiment["loss_val_{}".format(loss_name)].log(loss_value)
                    self.logger.experiment["epoch_log"].log(self.current_epoch)
                    self.logger.experiment["lr_log"].log(self.scheduler.get_last_lr())

                    self.logger.experiment["current_epoch_val"].log(self.current_epoch)
                    self.logger.experiment["global_step_val"].log(self.global_step)
                    self.logger.experiment["lr_val"].log(self.scheduler.get_last_lr())
                if self.text_logger is not None:
                    loss_text = ''
                    if val_loss_scalars:
                        loss_text = self._format_loss_scalars(val_loss_scalars) + ', '
                    str_to_print = (
                        f'val : epoch[{self.current_epoch:d}], steps[{self.global_step:d}] lr[{self.scheduler.get_last_lr()[0]:.6f}] | '
                        f'{loss_text}'
                        f'mIoU_val[{mIoU:.2f}], '
                        f'mAcc_val[{mAcc:.2f}], '
                        f'allAcc_val[{allAcc:.2f}], '
                    )
                    self.text_logger.loginfo(str_to_print)
                if val_loss_scalars:
                    self._append_loss_history(
                        'val',
                        self.current_epoch,
                        self.global_step,
                        float(self.scheduler.get_last_lr()[0]),
                        None,
                        val_loss_scalars)
                self._save_val_class_metrics(iou_class, accuracy_class)

            print("VAL: epoch[%d]: mIoU[%.3f] \n"%(self.current_epoch, mIoU))
            self.resetMetrics()

    @torch.no_grad()
    def test_step(self, data_dict, batch_idx):
        with torch.no_grad():
            outputs, losses = self.forward(data_dict)
            self.calcMetrics(batch_idx, outputs, data_dict, logName='test/accuracy')

            pred_idx = data_dict['pred_idx']
            pred_part = outputs['output']
            self.pred[pred_idx, :] += pred_part
            
    @torch.no_grad()
    def test_epoch_end(self, outputs):
        with torch.no_grad():
            iou_class = self.intersection_meter.sum / (self.union_meter.sum + 1e-10)
            accuracy_class = self.intersection_meter.sum / (self.target_meter.sum + 1e-10)
            mIoU = np.mean(iou_class)
            mAcc = np.mean(accuracy_class)
            allAcc = sum(self.intersection_meter.sum) / (sum(self.target_meter.sum) + 1e-10)
            
            if self.global_rank == 0:
                if self.logger is not None:
                    self.logger.experiment["mIoU_test_per_scene"].log(mIoU)
                    self.logger.experiment["mAcc_test_per_scene"].log(mAcc)
                    self.logger.experiment["allAcc_test_per_scene"].log(allAcc)

                    str_to_log = "TEST_per_scene: epoch[%d]: mIoU[%.3f], mAcc[%.3f], allAcc[%.3f] \n"%(self.current_epoch, mIoU, mAcc, allAcc)
                    print(str_to_log)
                    self.logger.experiment['logs'].log(str_to_log)
                if self.text_logger is not None:
                    str_to_print = (
                        f'test | '
                        f'mIoU_test_per_scene[{mIoU:.4f}], '
                        f'mAcc_test_per_scene[{mAcc:.4f}], '
                        f'allAcc_test_per_scene[{allAcc:.4f}], '
                    )
                    self.text_logger.loginfo(str_to_print)

        self.resetMetrics()

    def configure_optimizers(self):
        optimizers = []
        schedulers = []
        args = self.hparams['args']

        if self.optim in ['Adam', 'AdamW', 'NAdam'] :
            kwargs = \
                {
                    'params': self.parameters(),
                    'lr': float(args.lr), 
                    'weight_decay': args.weight_decay,
                }
            optimizer = getattr(torch.optim, self.optim)(**kwargs)
            milestones = [int(args.epochs*ratio) for ratio in args.schedule]
            for _ in range(5):
                print(">> lr schedule gamma[%f]"%(args.lr_GAMMA), milestones)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=args.lr_GAMMA)

        elif self.optim in ['SGD', 'ASGD']:
            optimizer = getattr(torch.optim, self.optim)(
                self.parameters(), lr=float(args.lr), momentum=args.momentum, weight_decay=args.weight_decay)
            milestones = [int(args.epochs*ratio) for ratio in args.schedule]
            for _ in range(5):
                print(">> lr schedule gamma[%f]"%(args.lr_GAMMA), milestones)
            self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, milestones=milestones, gamma=args.lr_GAMMA)
                
        else:
            raise NotImplemented
        optimizers.append(optimizer)

        return optimizers, schedulers

    # def train_dataloader(self):
    #     dataset = get_dataset(self.args.dataset)
    #     train_loader_kwargs = \
    #         {
    #             "batch_size": self.args.train_batch,
    #             "num_workers": self.args.train_worker,
    #             "collate_fn": dataset.TrainValCollateFn,
    #             "pin_memory": True,
    #             "drop_last": False,
    #             "shuffle": True,
    #         }
    #     train_loader = torch.utils.data.DataLoader(
    #         dataset.myImageFloder(self.args, mode=self.args.mode_train), **train_loader_kwargs)
    #     return train_loader
    
    # def val_dataloader(self):
    #     dataset = get_dataset(self.args.dataset)
    #     val_loader_kwargs = \
    #         {
    #             "batch_size": self.args.val_batch,
    #             "num_workers": self.args.val_worker,
    #             "collate_fn": dataset.TrainValCollateFn,
    #             "pin_memory": True,
    #             "drop_last": False,
    #             "shuffle": False,
    #         }
    #     val_loader = torch.utils.data.DataLoader(
    #         dataset.myImageFloder(self.args, mode=self.args.mode_eval), **val_loader_kwargs)
    #     return val_loader
