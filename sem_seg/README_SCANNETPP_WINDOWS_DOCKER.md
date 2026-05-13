# ScanNet++ training on Windows with Docker

This repo is Linux/CUDA-extension heavy, so the Windows path is Docker Desktop with WSL2 GPU support.

## 1. Check Docker GPU access

Run from PowerShell:

```powershell
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi
```

If this fails, start Docker Desktop, enable the WSL2 backend, and make sure Linux containers are active.

## 2. Start the PointMixer container

```powershell
docker pull jaesungchoe/pointmixer:cuda11.1
New-Item -ItemType Directory -Force E:\pointmixer_outputs

docker run -it --gpus all --name pointmixer-scannetpp --shm-size 12G `
  -v C:\Users\vladi\ECCV22-PointMixer-main:/code/ECCV22-PointMixer `
  -v E:\datasets:/datasets `
  -v E:\pointmixer_outputs:/workspace/outputs `
  jaesungchoe/pointmixer:cuda11.1
```

If the container already exists:

```powershell
docker start -ai pointmixer-scannetpp
```

## 3. Convert ScanNet++ to PointMixer format

Inside the container:

```bash
cd /code/ECCV22-PointMixer/sem_seg

python tools/prepare_scannetpp.py \
  --input-root /datasets/scannetpp_full/data \
  --output-root /datasets/pointmixer_scannetpp \
  --label-source benchmark
```

For only your current scene:

```bash
python tools/prepare_scannetpp.py \
  --input-root /datasets/scannetpp_full/data/00a231a370/scans \
  --output-root /datasets/pointmixer_scannetpp_00a231a370 \
  --label-source benchmark
```

With one scene, the converter writes it to both `train` and `val`. That is useful for smoke training and overfitting checks, but it is not a real validation setup.

## 4. Train on RTX 3060 12GB

Inside the container:

```bash
cd /code/ECCV22-PointMixer/sem_seg

SCANNETPP_ROOT=/datasets/pointmixer_scannetpp \
SAVEROOT=/workspace/outputs/PointMixerScanNetPP \
EPOCHS=10 \
TRAIN_VOXEL_MAX=16000 \
EVAL_VOXEL_MAX=30000 \
WORKERS=0 \
bash script/run_scannetpp_PointMixer_3060.sh
```

For the single-scene smoke dataset:

```bash
SCANNETPP_ROOT=/datasets/pointmixer_scannetpp_00a231a370 \
SAVEROOT=/workspace/outputs/PointMixerScanNetPP_00a231a370 \
EPOCHS=5 \
TRAIN_VOXEL_MAX=12000 \
EVAL_VOXEL_MAX=20000 \
WORKERS=0 \
bash script/run_scannetpp_PointMixer_3060.sh
```

## Memory knobs

Start conservative on 12GB VRAM and 16GB RAM:

- `TRAIN_VOXEL_MAX=12000` if CUDA OOM happens.
- `EVAL_VOXEL_MAX=20000` if validation OOM happens.
- `WORKERS=0` or `WORKERS=1` for Windows Docker mounts.
- `NUM_TRAIN_BATCH=1` should stay at 1 on 12GB.
- `--shm-size 12G` is intentionally below 16GB system RAM.

## Outputs

Checkpoints and logs are written under the mounted Windows folder:

```text
E:\pointmixer_outputs
```
