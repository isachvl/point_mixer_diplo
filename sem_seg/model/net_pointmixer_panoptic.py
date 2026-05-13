# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .net_pointmixer import net_pointmixer


def _lovasz_grad(gt_sorted):
    """Gradient of the Lovasz extension with respect to sorted errors."""
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1.0 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union.clamp(min=1e-6)
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def _flatten_probs(probs, labels, ignore_label):
    valid = labels != ignore_label
    if valid.sum() == 0:
        return probs.new_zeros((0, probs.shape[1])), labels.new_zeros((0,), dtype=torch.long)
    return probs[valid], labels[valid]


def lovasz_softmax_loss(logits, labels, ignore_label):
    """Lovasz-Softmax loss for point-wise multi-class IoU optimization."""
    probs = F.softmax(logits, dim=1)
    probs, labels = _flatten_probs(probs, labels, ignore_label)
    if probs.numel() == 0:
        return logits.sum() * 0.0

    losses = []
    classes = torch.unique(labels)
    for class_idx in classes:
        class_id = int(class_idx.item())
        fg = (labels == class_id).float()
        if fg.sum() == 0:
            continue
        class_pred = probs[:, class_id]
        errors = (fg - class_pred).abs()
        errors_sorted, perm = torch.sort(errors, descending=True)
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def focal_loss(logits, labels, ignore_label, gamma=2.0, class_weight=None):
    """Multi-class focal loss for hard/rare point labels."""
    valid = labels != ignore_label
    if valid.sum() == 0:
        return logits.sum() * 0.0
    logits = logits[valid]
    labels = labels[valid]
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, labels.view(-1, 1)).squeeze(1)
    pt = log_pt.exp().clamp(min=1e-6, max=1.0)
    loss = -((1.0 - pt) ** float(gamma)) * log_pt
    if class_weight is not None:
        loss = loss * class_weight[labels]
    return loss.mean()


class net_pointmixer_panoptic(net_pointmixer):
    """Lightning-обертка для обучения PointMixerPanoptic.

    Сама архитектура лежит в model/network/pointmixer.py, а здесь описано,
    какие выходы модели брать и какие loss-функции считать для обучения.
    """

    def __init__(self, args=None):
        super().__init__(args=args)
        args = self.hparams['args']
        self.train_offset_only = bool(getattr(args, 'train_offset_only', False))
        self.semantic_loss_type = str(
            getattr(args, 'semantic_loss_type', 'ce') or 'ce').lower()
        self.focal_gamma = float(getattr(args, 'focal_gamma', 2.0) or 2.0)
        self.focal_loss_weight = float(
            getattr(args, 'focal_loss_weight', 1.0) or 1.0)
        self.lovasz_loss_weight = float(
            getattr(args, 'lovasz_loss_weight', 0.0) or 0.0)
        self.semantic_label_smoothing = float(
            getattr(args, 'semantic_label_smoothing', 0.0) or 0.0)
        weight_path = getattr(args, 'class_weight_path', None)
        if weight_path:
            # Веса классов нужны из-за дисбаланса датасета:
            # редкие классы получают больший вклад в semantic loss.
            if weight_path.endswith('.pt') or weight_path.endswith('.pth'):
                weight = torch.load(weight_path, map_location='cpu')
                if isinstance(weight, dict):
                    weight = weight.get('weights', weight.get('class_weights'))
            else:
                weight = np.load(weight_path)
            weight = torch.as_tensor(weight, dtype=torch.float32)
            if weight.numel() != self.classes:
                raise ValueError(
                    'class_weight_path has {} weights, expected {}'.format(
                        weight.numel(), self.classes))
            self.register_buffer('semantic_class_weight', weight.view(-1))
            if self.global_rank == 0:
                print('[PM PANOPTIC] loaded class weights:', weight_path)
        else:
            self.semantic_class_weight = None

        if self.train_offset_only:
            self._freeze_for_offset_only()

    def _freeze_for_offset_only(self):
        """Freeze everything except the offset head for careful panoptic fine-tuning."""
        for name, param in self.model.named_parameters():
            param.requires_grad = name.startswith('offset.')
        if self.global_rank == 0:
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print('[PM PANOPTIC] TRAIN_OFFSET_ONLY=True: trainable params {}/{}'.format(
                trainable, total))

    def _keep_frozen_parts_eval(self):
        if not self.train_offset_only:
            return
        # BatchNorm in the frozen backbone/semantic head must not drift during offset fine-tuning.
        self.model.eval()
        self.model.offset.train(self.training)

    def _semantic_loss(self, semantic_logits, target):
        semantic_weight = self.semantic_class_weight
        ce_kwargs = {
            'weight': semantic_weight,
            'ignore_index': self.ignore_label,
        }
        if self.semantic_label_smoothing > 0:
            ce_kwargs['label_smoothing'] = self.semantic_label_smoothing

        if self.semantic_loss_type in ('ce', 'cross_entropy'):
            return F.cross_entropy(semantic_logits, target, **ce_kwargs)
        if self.semantic_loss_type in ('focal', 'focal_loss'):
            return focal_loss(
                semantic_logits, target, self.ignore_label,
                gamma=self.focal_gamma,
                class_weight=semantic_weight)
        if self.semantic_loss_type in ('lovasz', 'lovasz_softmax'):
            return lovasz_softmax_loss(semantic_logits, target, self.ignore_label)
        if self.semantic_loss_type in ('focal_lovasz', 'focal+lovasz', 'focal_lovasz_loss'):
            focal = focal_loss(
                semantic_logits, target, self.ignore_label,
                gamma=self.focal_gamma,
                class_weight=semantic_weight)
            lovasz = lovasz_softmax_loss(semantic_logits, target, self.ignore_label)
            return self.focal_loss_weight * focal + self.lovasz_loss_weight * lovasz
        raise ValueError('Unknown semantic_loss_type: {}'.format(self.semantic_loss_type))

    def forward(self, data_dict):
        pred_dict = {}
        loss_dict = {}

        with torch.no_grad():
            # Данные из panoptic loader:
            # coord/feat - точки и признаки, target - semantic label,
            # target_instance/target_offset - разметка для instance-части.
            coord = data_dict['coord']
            feat = data_dict['feat']
            target = data_dict['target']
            offset = data_dict['offset']
            target_instance = data_dict.get('target_instance')
            target_offset = data_dict.get('target_offset')

        self._keep_frozen_parts_eval()
        if self.train_offset_only:
            with torch.no_grad():
                point_features = self.model.forward_features([coord, feat, offset])
                semantic_logits = self.model.cls(point_features)
            pred_offsets = self.model.offset(point_features.detach())
            outputs = {
                'semantic_logits': semantic_logits,
                'pt_offsets': pred_offsets,
            }
        else:
            # Вызов самой архитектуры PointMixerPanoptic.
            # Она возвращает semantic logits и offset для каждой точки.
            outputs = self.model([coord, feat, offset])
        semantic_logits = outputs['semantic_logits']
        pred_offsets = outputs['pt_offsets']

        # pred_dict нужен базовому validation-коду из net_pointmixer:
        # semantic_logits используются для mIoU/mAcc/allAcc.
        pred_dict['output'] = semantic_logits
        pred_dict['pt_offsets'] = pred_offsets

        if target.shape[-1] == 1:
            target = target[:, 0]

        # Semantic loss учит модель отвечать "какой класс у точки".
        semantic_loss = self._semantic_loss(semantic_logits, target)
        loss_dict['semantic_loss'] = semantic_loss

        if target_instance is None or target_offset is None:
            # Если в датасете нет instance/offset-разметки, оставляем только
            # semantic loss, а offset loss делаем нулевым без поломки графа.
            zero = pred_offsets.sum() * 0.0
            loss_dict['offset_norm_loss'] = zero
            loss_dict['offset_dir_loss'] = zero
            return pred_dict, loss_dict

        valid = (target != self.ignore_label) & (target_instance >= 0)
        if torch.any(valid):
            # Offset loss учит точку двигаться к центру своего объекта.
            pred_valid = pred_offsets[valid]
            target_valid = target_offset[valid]
            offset_norm_loss = torch.norm(pred_valid - target_valid, p=1, dim=1).mean()

            # Direction loss дополнительно выравнивает направление offset-вектора.
            pred_dir = F.normalize(pred_valid, p=2, dim=1)
            target_dir = F.normalize(target_valid, p=2, dim=1)
            offset_dir_loss = (1.0 - torch.sum(pred_dir * target_dir, dim=1)).mean()
        else:
            offset_norm_loss = pred_offsets.sum() * 0.0
            offset_dir_loss = pred_offsets.sum() * 0.0

        args = self.hparams['args']
        # Важно: если в команде OFFSET_LOSS_WEIGHT=0 и OFFSET_DIR_LOSS_WEIGHT=0,
        # архитектура остается паноптической, но offset-ветка фактически не учится.
        loss_dict['offset_norm_loss'] = float(args.offset_loss_weight) * offset_norm_loss
        loss_dict['offset_dir_loss'] = float(args.offset_dir_loss_weight) * offset_dir_loss
        return pred_dict, loss_dict
