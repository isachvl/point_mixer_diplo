# PointMixerPanoptic: что запускать и где находится архитектура

Этот файл нужен как короткая памятка по нашей версии PointMixerPanoptic.
Датасет и чекпоинты сюда не входят: они лежат отдельно в `E:\datasets` и
`E:\pointmixer_outputs`, а в репозитории остается только код архитектуры,
обучения и запуска.

## Главные файлы

1. `sem_seg/script/run_scannetpp_PointMixerPanoptic_3060.sh`
   - основной bash-скрипт обучения внутри Docker;
   - задает `ARCH="pointmixer_panoptic"`;
   - задает `MODEL="net_pointmixer_panoptic"`;
   - задает `DATASET="loader_scannetpp_panoptic"`;
   - в конце запускает `python train_pl.py`.

2. `sem_seg/train_pl.py`
   - главный Python-файл обучения;
   - создает dataloader;
   - создает модель;
   - запускает `trainer.fit(...)`.

3. `sem_seg/model/network/pointmixer.py`
   - сама архитектура PointMixer;
   - `PointMixerSegNet` - обычная семантическая сегментация;
   - `PointMixerPanopticNet` - panoptic-версия с двумя выходами:
     `semantic_logits` и `pt_offsets`.

4. `sem_seg/model/net_pointmixer_panoptic.py`
   - обучающая Lightning-обертка;
   - вызывает архитектуру из `pointmixer.py`;
   - считает `semantic_loss`;
   - считает `offset_norm_loss` и `offset_dir_loss`, если они включены.

5. `sem_seg/dataset/loader_scannetpp_panoptic.py`
   - загружает подготовленные `.pth` сцены;
   - возвращает точки, признаки, semantic labels, instance labels;
   - строит target offsets к центрам объектов.

## Как идет запуск

Цепочка запуска такая:

```text
PowerShell команда / .ps1 файл
        ↓
docker exec pointmixer-scannetpp
        ↓
sem_seg/script/run_scannetpp_PointMixerPanoptic_3060.sh
        ↓
sem_seg/train_pl.py
        ↓
sem_seg/model/net_pointmixer_panoptic.py
        ↓
sem_seg/model/network/pointmixer.py
```

По смыслу:

```text
run_scannetpp_PointMixerPanoptic_3060.sh
выбирает архитектуру и параметры

train_pl.py
создает dataset, dataloader, model и trainer

net_pointmixer_panoptic.py
управляет обучением panoptic-модели и считает loss

pointmixer.py
делает прямой проход нейросети
```

## Что такое PointMixerPanoptic

Упрощенно:

```text
PointMixerPanoptic = PointMixer backbone + semantic head + offset head
```

Модель выдает:

```text
semantic_logits - класс каждой точки
pt_offsets      - сдвиг точки к центру своего объекта
```

Паноптика получается так:

```text
точка -> класс объекта/поверхности -> offset к центру -> кластеризация -> instance_id
```

Важный нюанс: если запускать с параметрами
`OFFSET_LOSS_WEIGHT=0` и `OFFSET_DIR_LOSS_WEIGHT=0`, то архитектура остается
паноптической, но offset-ветка фактически не обучается. В таком режиме модель
в основном улучшает semantic segmentation, а instance-разделение потом
получается за счет кластеризации на inference.

## Команда fine-tune от лучшего чекпоинта

Windows PowerShell:

```powershell
docker exec -it pointmixer-scannetpp bash -lc "cd /code/ECCV22-PointMixer/sem_seg && SCANNETPP_ROOT=/datasets/pointmixer_scannetpp_top100_panoptic_rgb_pv004 SAVEROOT=/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_pv004_finetune_ep20_vmax16000_lr3e5 EPOCHS=12 LOOP=10 LR=0.00003 OPTIM=AdamW VOX_SIZE=0 TRAIN_VOXEL_MAX=16000 EVAL_VOXEL_MAX=50000 WORKERS=1 BLOCK_SIZE=3.0 MIN_POINTS_IN_BLOCK=1024 MIN_TRAIN_POINTS=4096 CLASS_BALANCE_PROB=0.25 OFFSET_LOSS_WEIGHT=0 OFFSET_DIR_LOSS_WEIGHT=0 AUTO_CLASS_WEIGHTS=True CLASS_WEIGHT_POWER=0.5 CLASS_WEIGHT_MAX=5 LOAD_MODEL=/workspace/outputs/PointMixerScanNetPP_panoptic_rgb_pv004_block_balanced/2026-05-02_19-40-27__scannetpp__pointmixer_panoptic_3060/epoch=020--mIoU_val=0.1720--.ckpt bash script/run_scannetpp_PointMixerPanoptic_3060.sh"
```

То же самое лежит в отдельном файле:

```text
sem_seg/script/run_scannetpp_panoptic_finetune_ep20_vmax16000_lr3e5.ps1
```

## Что значат основные параметры

```text
SCANNETPP_ROOT       - путь к подготовленному датасету внутри Docker
SAVEROOT             - куда сохранять новые чекпоинты и логи
LOAD_MODEL           - от какого чекпоинта начинать fine-tune
EPOCHS               - сколько эпох дообучать
LOOP                 - сколько раз прокручивать train-сцены за эпоху
LR                   - learning rate
VOX_SIZE=0           - не делать дополнительное voxel-сжатие во время обучения
TRAIN_VOXEL_MAX      - максимум точек в train-crop
EVAL_VOXEL_MAX       - максимум точек на validation
BLOCK_SIZE           - размер локального куска сцены по XY
CLASS_BALANCE_PROB   - вероятность выбрать crop вокруг редкого класса
OFFSET_LOSS_WEIGHT   - вес loss для длины offset-вектора
OFFSET_DIR_LOSS_WEIGHT - вес loss для направления offset-вектора
```

Если будет CUDA OOM, сначала уменьшить:

```text
TRAIN_VOXEL_MAX=12000
EVAL_VOXEL_MAX=40000
```

