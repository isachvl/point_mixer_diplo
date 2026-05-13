#!/bin/bash
set -euo pipefail

### Paths inside Docker.
SCANNETPP_ROOT=${SCANNETPP_ROOT:-/datasets/pointmixer_scannetpp}
SAVEROOT=${SAVEROOT:-/workspace/outputs/PointMixerScanNetPP}

### 3060 12GB / 16GB RAM friendly defaults.
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MASTER_ADDR=${MASTER_ADDR:-localhost}
export NODE_RANK=${NODE_RANK:-0}

WORKERS=${WORKERS:-0}
NUM_GPUS=${NUM_GPUS:-1}
NUM_TRAIN_BATCH=${NUM_TRAIN_BATCH:-1}
NUM_VAL_BATCH=${NUM_VAL_BATCH:-1}
NUM_TEST_BATCH=${NUM_TEST_BATCH:-1}
TRAIN_VOXEL_MAX=${TRAIN_VOXEL_MAX:-16000}
EVAL_VOXEL_MAX=${EVAL_VOXEL_MAX:-30000}
VOX_SIZE=${VOX_SIZE:-0.05}
LOOP=${LOOP:-20}
EPOCHS=${EPOCHS:-10}
LR=${LR:-0.02}
NSAMPLE=${NSAMPLE:-"8 8 8 8 8"}

ARCH="pointmixer"
DATASET="loader_scannetpp"
INTRALAYER="PointMixerIntraSetLayer"
INTERLAYER="PointMixerInterSetLayer"
TRANSDOWN="SymmetricTransitionDownBlock"
TRANSUP="SymmetricTransitionUpBlock"

if [ ! -f "${SCANNETPP_ROOT}/meta/classes.txt" ]; then
  echo "[PM ERROR] ${SCANNETPP_ROOT}/meta/classes.txt not found."
  echo "[PM ERROR] Run tools/prepare_scannetpp.py before training."
  exit 1
fi

CLASSES=${CLASSES:-$(cat "${SCANNETPP_ROOT}/meta/classes.txt")}
DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
COMPUTER="RTX3060-12GB-Windows-Docker"
MYCHECKPOINT="${SAVEROOT}/${DATE_TIME}__scannetpp__pointmixer_3060/"

mkdir -p "${MYCHECKPOINT}"
cp -a "model" "${MYCHECKPOINT}/model"
cp -a "script/$(basename "$0")" "${MYCHECKPOINT}/$(basename "$0")"

echo "[PM INFO] Building CUDA pointops if needed..."
bash env_setup.sh

echo "[PM INFO] Training ScanNet++ PointMixer with ${CLASSES} classes."
python train_pl.py \
  --MYCHECKPOINT "${MYCHECKPOINT}" --computer "${COMPUTER}" --shell "$(basename "$0")" \
  --MASTER_ADDR "${MASTER_ADDR}" \
  --train_worker "${WORKERS}" --val_worker "${WORKERS}" \
  --NUM_GPUS "${NUM_GPUS}" \
  --train_batch "${NUM_TRAIN_BATCH}" \
  --val_batch "${NUM_VAL_BATCH}" \
  --test_batch "${NUM_TEST_BATCH}" \
  --scannetpp_root "${SCANNETPP_ROOT}" \
  --neptune_proj "local/pointmixer-scannetpp" \
  --epochs "${EPOCHS}" --check_val_every_n_epoch 1 --lr "${LR}" \
  --dataset "${DATASET}" --optim "SGD" \
  --model "net_pointmixer" --arch "${ARCH}" \
  --intraLayer "${INTRALAYER}" --interLayer "${INTERLAYER}" \
  --transdown "${TRANSDOWN}" --transup "${TRANSUP}" \
  --nsample ${NSAMPLE} --drop_rate 0.1 --fea_dim 6 --classes "${CLASSES}" --loop "${LOOP}" \
  --voxel_size "${VOX_SIZE}" --train_voxel_max "${TRAIN_VOXEL_MAX}" --eval_voxel_max "${EVAL_VOXEL_MAX}" \
  --mode_train "train" --mode_eval "val" --aug "scannetpp" \
  --cudnn_benchmark False
