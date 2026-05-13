#!/bin/bash
set -euo pipefail

### Paths inside Docker.
SCANNETPP_ROOT=${SCANNETPP_ROOT:-/datasets/pointmixer_scannetpp_top100_panoptic}
SAVEROOT=${SAVEROOT:-/workspace/outputs/PointMixerScanNetPP_panoptic}

### RTX 3060 12GB / 16GB RAM friendly defaults.
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
LOOP=${LOOP:-8}
EPOCHS=${EPOCHS:-10}
LR=${LR:-0.005}
OPTIM=${OPTIM:-SGD}
FEA_DIM=${FEA_DIM:-6}
NSAMPLE=${NSAMPLE:-"8 8 8 8 8"}
BLOCK_SIZE=${BLOCK_SIZE:-0}
MIN_POINTS_IN_BLOCK=${MIN_POINTS_IN_BLOCK:-1024}
MIN_TRAIN_POINTS=${MIN_TRAIN_POINTS:-0}
CLASS_BALANCE_PROB=${CLASS_BALANCE_PROB:-0}
OFFSET_LOSS_WEIGHT=${OFFSET_LOSS_WEIGHT:-1.0}
OFFSET_DIR_LOSS_WEIGHT=${OFFSET_DIR_LOSS_WEIGHT:-0.2}
TRAIN_OFFSET_ONLY=${TRAIN_OFFSET_ONLY:-False}
LOAD_MODEL=${LOAD_MODEL:-}
STRICT_LOAD=${STRICT_LOAD:-True}
RESUME=${RESUME:-False}
AUTO_CLASS_WEIGHTS=${AUTO_CLASS_WEIGHTS:-False}
CLASS_WEIGHT_PATH=${CLASS_WEIGHT_PATH:-}
CLASS_WEIGHT_POWER=${CLASS_WEIGHT_POWER:-0.5}
CLASS_WEIGHT_MIN=${CLASS_WEIGHT_MIN:-0.25}
CLASS_WEIGHT_MAX=${CLASS_WEIGHT_MAX:-5.0}
SEMANTIC_LABEL_SMOOTHING=${SEMANTIC_LABEL_SMOOTHING:-0.0}
SEMANTIC_LOSS_TYPE=${SEMANTIC_LOSS_TYPE:-ce}
FOCAL_GAMMA=${FOCAL_GAMMA:-2.0}
FOCAL_LOSS_WEIGHT=${FOCAL_LOSS_WEIGHT:-1.0}
LOVASZ_LOSS_WEIGHT=${LOVASZ_LOSS_WEIGHT:-0.0}

ARCH="pointmixer_panoptic"
MODEL="net_pointmixer_panoptic"
DATASET="loader_scannetpp_panoptic"
INTRALAYER="PointMixerIntraSetLayer"
INTERLAYER="PointMixerInterSetLayer"
TRANSDOWN="SymmetricTransitionDownBlock"
TRANSUP="SymmetricTransitionUpBlock"

if [ ! -f "${SCANNETPP_ROOT}/meta/classes.txt" ]; then
  echo "[PM ERROR] ${SCANNETPP_ROOT}/meta/classes.txt not found."
  echo "[PM ERROR] Run tools/prepare_scannetpp_panoptic.py before training."
  exit 1
fi

CLASSES=${CLASSES:-$(cat "${SCANNETPP_ROOT}/meta/classes.txt")}
DATE_TIME=$(date +"%Y-%m-%d_%H-%M-%S")
COMPUTER="RTX3060-12GB-Windows-Docker"
MYCHECKPOINT="${SAVEROOT}/${DATE_TIME}__scannetpp__pointmixer_panoptic_3060/"
SCRIPT_NAME=$(basename "$0")

mkdir -p "${MYCHECKPOINT}"
cp -a "model" "${MYCHECKPOINT}/model"
cp -a "script/${SCRIPT_NAME}" "${MYCHECKPOINT}/${SCRIPT_NAME}"

echo "[PM INFO] Building CUDA pointops if needed..."
bash env_setup.sh

if [ "${AUTO_CLASS_WEIGHTS}" = "True" ] || [ "${AUTO_CLASS_WEIGHTS}" = "true" ] || [ "${AUTO_CLASS_WEIGHTS}" = "1" ]; then
  if [ -z "${CLASS_WEIGHT_PATH}" ]; then
    CLASS_WEIGHT_PATH="${SCANNETPP_ROOT}/meta/class_weights_p${CLASS_WEIGHT_POWER}_max${CLASS_WEIGHT_MAX}.npy"
  fi
  if [ ! -f "${CLASS_WEIGHT_PATH}" ]; then
    echo "[PM INFO] Computing class weights: ${CLASS_WEIGHT_PATH}"
    python tools/compute_scannetpp_class_weights.py \
      --data-root "${SCANNETPP_ROOT}" \
      --output "${CLASS_WEIGHT_PATH}" \
      --classes "${CLASSES}" \
      --power "${CLASS_WEIGHT_POWER}" \
      --min-weight "${CLASS_WEIGHT_MIN}" \
      --max-weight "${CLASS_WEIGHT_MAX}"
  fi
fi

EXTRA_ARGS=()
if [ -n "${LOAD_MODEL}" ]; then
  EXTRA_ARGS+=(--load_model "${LOAD_MODEL}" --strict_load "${STRICT_LOAD}")
fi
if [ "${RESUME}" = "True" ] || [ "${RESUME}" = "true" ] || [ "${RESUME}" = "1" ]; then
  EXTRA_ARGS+=(--resume)
fi
if [ -n "${CLASS_WEIGHT_PATH}" ]; then
  EXTRA_ARGS+=(--class_weight_path "${CLASS_WEIGHT_PATH}")
fi
if [ "${SEMANTIC_LABEL_SMOOTHING}" != "0" ] && [ "${SEMANTIC_LABEL_SMOOTHING}" != "0.0" ]; then
  EXTRA_ARGS+=(--semantic_label_smoothing "${SEMANTIC_LABEL_SMOOTHING}")
fi

echo "[PM INFO] Training ScanNet++ PointMixer Panoptic with ${CLASSES} classes."
python train_pl.py \
  --MYCHECKPOINT "${MYCHECKPOINT}" --computer "${COMPUTER}" --shell "${SCRIPT_NAME}" \
  --MASTER_ADDR "${MASTER_ADDR}" \
  --train_worker "${WORKERS}" --val_worker "${WORKERS}" \
  --NUM_GPUS "${NUM_GPUS}" \
  --train_batch "${NUM_TRAIN_BATCH}" \
  --val_batch "${NUM_VAL_BATCH}" \
  --test_batch "${NUM_TEST_BATCH}" \
  --scannetpp_root "${SCANNETPP_ROOT}" \
  --neptune_proj "local/pointmixer-scannetpp-panoptic" \
  --epochs "${EPOCHS}" --check_val_every_n_epoch 1 --lr "${LR}" \
  --dataset "${DATASET}" --optim "${OPTIM}" \
  --model "${MODEL}" --arch "${ARCH}" \
  --intraLayer "${INTRALAYER}" --interLayer "${INTERLAYER}" \
  --transdown "${TRANSDOWN}" --transup "${TRANSUP}" \
  --nsample ${NSAMPLE} --drop_rate 0.1 --fea_dim "${FEA_DIM}" --classes "${CLASSES}" --loop "${LOOP}" \
  --voxel_size "${VOX_SIZE}" --train_voxel_max "${TRAIN_VOXEL_MAX}" --eval_voxel_max "${EVAL_VOXEL_MAX}" \
  --block_size "${BLOCK_SIZE}" --min_points_in_block "${MIN_POINTS_IN_BLOCK}" \
  --min_train_points "${MIN_TRAIN_POINTS}" \
  --class_balance_prob "${CLASS_BALANCE_PROB}" \
  --mode_train "train" --mode_eval "val" --aug "scannetpp" \
  --offset_loss_weight "${OFFSET_LOSS_WEIGHT}" --offset_dir_loss_weight "${OFFSET_DIR_LOSS_WEIGHT}" \
  --train_offset_only "${TRAIN_OFFSET_ONLY}" \
  --semantic_loss_type "${SEMANTIC_LOSS_TYPE}" \
  --focal_gamma "${FOCAL_GAMMA}" \
  --focal_loss_weight "${FOCAL_LOSS_WEIGHT}" \
  --lovasz_loss_weight "${LOVASZ_LOSS_WEIGHT}" \
  --cudnn_benchmark False \
  "${EXTRA_ARGS[@]}"
