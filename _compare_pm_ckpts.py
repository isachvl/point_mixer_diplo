import torch
from pathlib import Path

runs = {
    'pv004_ce_best_0201': '/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_pv004_finetune_ep20_vmax16000_lr3e5/2026-05-03_17-20-24__scannetpp__pointmixer_panoptic_3060/epoch=000--mIoU_val=0.2010--.ckpt',
    'pv004_focal_lovasz_best_01993': '/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_pv004_focal_lovasz_ep10_continue_to_miou020/2026-05-06_17-54-56__scannetpp__pointmixer_panoptic_3060/epoch=008--mIoU_val=0.1993--.ckpt',
    'pv003_focal_lovasz_best_01759': '/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_pv003_focal_lovasz_ep20_resume/2026-05-10_17-47-13__scannetpp__pointmixer_panoptic_3060/epoch=014--mIoU_val=0.1759--.ckpt',
    'dense005_focal_lovasz_best_bad': '/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_dense005_focal_lovasz_ep20/2026-05-08_19-31-36__scannetpp__pointmixer_panoptic_3060/epoch=000--mIoU_val=0.0070--.ckpt',
    'dense005_ce_adapt_best_bad': '/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_dense005_adapt_from_miou0201_ce_ep5/2026-05-09_06-02-13__scannetpp__pointmixer_panoptic_3060/epoch=003--mIoU_val=0.0180--.ckpt',
}
keys = [
    'scannetpp_root','epochs','loop','lr','optim','voxel_size','train_voxel_max','eval_voxel_max',
    'block_size','min_points_in_block','min_train_points','class_balance_prob',
    'semantic_loss_type','focal_gamma','focal_loss_weight','lovasz_loss_weight',
    'class_weight_path','semantic_label_smoothing','offset_loss_weight','offset_dir_loss_weight',
    'train_offset_only','load_model','resume','strict_load','classes','train_worker','val_worker'
]
for name, path in runs.items():
    print('\n###', name)
    ckpt = torch.load(path, map_location='cpu')
    hp = ckpt.get('hyper_parameters', {}) or {}
    args = hp.get('args', hp)
    if not isinstance(args, dict):
        args = vars(args)
    print('path:', path)
    print('epoch:', ckpt.get('epoch'), 'global_step:', ckpt.get('global_step'))
    for k in keys:
        print(f'{k}: {args.get(k)}')
