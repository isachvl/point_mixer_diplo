# PointMixer Panoptic для ScanNet++ на Windows/Docker

Эта ветка добавляет PointGroup-style panoptic режим без OneFormer3D/Minkowski:

- PointMixer backbone
- semantic head
- offset head к центру объекта
- inference-кластеризация по `xyz + predicted_offset`
- выходы: цветной PLY, `class_label`, `instance_id`, `.npy`, `.csv`, `.xlsx`

## 1. Подготовить panoptic датасет

```powershell
docker exec -it pointmixer-scannetpp bash -lc "cd /code/ECCV22-PointMixer/sem_seg && python tools/prepare_scannetpp_panoptic.py --input-root /datasets/scannetpp_full/data --output-root /datasets/pointmixer_scannetpp_top100_panoptic --label-source benchmark"
```

По умолчанию `wall`, `floor`, `ceiling` остаются semantic-классами, но не получают instance/offset цели.

## 2. Обучить PointMixer Panoptic

```powershell
docker exec -it pointmixer-scannetpp bash -lc "cd /code/ECCV22-PointMixer/sem_seg && SCANNETPP_ROOT=/datasets/pointmixer_scannetpp_top100_panoptic SAVEROOT=/workspace/outputs/PointMixerScanNetPP_panoptic EPOCHS=10 LOOP=8 LR=0.005 TRAIN_VOXEL_MAX=16000 EVAL_VOXEL_MAX=30000 WORKERS=0 bash script/run_scannetpp_PointMixerPanoptic_3060.sh"
```

Для RTX 3060 12GB и 16GB RAM безопаснее начинать с `WORKERS=0`.

## 2a. Ускоренный вариант, если GPU простаивает

Полный датасет читает огромные `.pth` сцены на каждом шаге, поэтому на HDD/Windows Docker GPU может ждать CPU и диск. Быстрее подготовить отдельный prevoxel-датасет один раз:

```powershell
docker exec -it pointmixer-scannetpp bash -lc "cd /code/ECCV22-PointMixer/sem_seg && python tools/prepare_scannetpp_panoptic.py --input-root /datasets/scannetpp_full/data --output-root /datasets/pointmixer_scannetpp_top100_panoptic_fast008 --label-source benchmark --prevoxel-size 0.08"
```

Потом учить с выключенным повторным voxelize:

```powershell
docker exec -it pointmixer-scannetpp bash -lc "cd /code/ECCV22-PointMixer/sem_seg && SCANNETPP_ROOT=/datasets/pointmixer_scannetpp_top100_panoptic_fast008 SAVEROOT=/workspace/outputs/PointMixerScanNetPP_panoptic_fast008 EPOCHS=10 LOOP=8 LR=0.005 VOX_SIZE=0 TRAIN_VOXEL_MAX=24000 EVAL_VOXEL_MAX=30000 WORKERS=1 bash script/run_scannetpp_PointMixerPanoptic_3060.sh"
```

Если Docker Desktop убивает worker по памяти, замени `WORKERS=1` на `WORKERS=0`.

## 3. Запустить panoptic inference на одной сцене

Замени `<CKPT>` на лучший `.ckpt` из папки обучения.

```powershell
docker exec -it pointmixer-scannetpp bash -lc "cd /code/ECCV22-PointMixer/sem_seg && python tools/infer_scannetpp_panoptic_scene.py --scene-ply /datasets/облака+разметка/data/1a130d092a/scans/pc_aligned.ply --checkpoint <CKPT> --label-map /datasets/pointmixer_scannetpp_top100_panoptic/meta/label_mapping.tsv --output-dir /workspace/outputs/PointMixerPanopticPredictions/1a130d092a --max-points 16000 --block-size 2.0 --votes 1 --input-voxel-size 0.05 --confidence-threshold 0.35 --cluster-radius 0.12 --min-cluster-points 50 --write-point-csv"
```

Основные параметры inference:

- `--confidence-threshold 0.35` убирает неуверенные точки.
- `--cluster-radius 0.12` радиус склейки точек одного объекта после offset-сдвига.
- `--min-cluster-points 50` мелкие шумовые кластеры не считаются объектами.
- `--input-voxel-size 0.05` сильно ускоряет плотные `pc_aligned.ply`.

Выходные файлы:

- `*_pointmixer_panoptic_colored.ply`
- `*_pointmixer_panoptic_labels.npy`, где колонки: `class_label`, `instance_id`
- `*_pointmixer_panoptic_instance_summary.xlsx`
- `*_pointmixer_panoptic_instance_summary.csv`
- `*_pointmixer_panoptic_points.csv`, если включен `--write-point-csv`
