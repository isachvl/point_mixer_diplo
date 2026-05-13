# Запуск fine-tune PointMixerPanoptic из Windows PowerShell.
# Датасет и чекпоинты должны быть примонтированы в Docker-контейнер
# pointmixer-scannetpp как /datasets и /workspace/outputs.

$ErrorActionPreference = "Stop"

$Container = "pointmixer-scannetpp"
$DatasetRoot = "/datasets/pointmixer_scannetpp_top100_panoptic_rgb_pv004"
$SaveRoot = "/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_pv004_finetune_ep20_vmax16000_lr3e5"
$Checkpoint = "/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_pv004_block_balanced/2026-05-02_19-40-27__scannetpp__pointmixer_panoptic_3060/epoch=020--mIoU_val=0.1720--.ckpt"

# Важно: RESUME=True здесь не ставим.
# Это именно fine-tune от лучших весов epoch 20 с новым optimizer/LR.
$InnerCommand = "cd /code/ECCV22-PointMixer/sem_seg && " +
    "SCANNETPP_ROOT=$DatasetRoot " +
    "SAVEROOT=$SaveRoot " +
    "EPOCHS=12 " +
    "LOOP=10 " +
    "LR=0.00003 " +
    "OPTIM=AdamW " +
    "VOX_SIZE=0 " +
    "TRAIN_VOXEL_MAX=16000 " +
    "EVAL_VOXEL_MAX=50000 " +
    "WORKERS=1 " +
    "BLOCK_SIZE=3.0 " +
    "MIN_POINTS_IN_BLOCK=1024 " +
    "MIN_TRAIN_POINTS=4096 " +
    "CLASS_BALANCE_PROB=0.25 " +
    "OFFSET_LOSS_WEIGHT=0 " +
    "OFFSET_DIR_LOSS_WEIGHT=0 " +
    "AUTO_CLASS_WEIGHTS=True " +
    "CLASS_WEIGHT_POWER=0.5 " +
    "CLASS_WEIGHT_MAX=5 " +
    "LOAD_MODEL=$Checkpoint " +
    "bash script/run_scannetpp_PointMixerPanoptic_3060.sh"

docker exec -it $Container bash -lc $InnerCommand
