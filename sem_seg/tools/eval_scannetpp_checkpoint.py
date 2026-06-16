from __future__ import print_function

import inspect
import os
import random
import sys

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from colorama import Fore, Style

try:
    from pytorch_lightning.strategies import DDPStrategy
except ModuleNotFoundError:
    DDPStrategy = None

try:
    from pytorch_lightning.plugins import DDPPlugin
except (ImportError, ModuleNotFoundError):
    DDPPlugin = None

SEM_SEG_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if SEM_SEG_ROOT not in sys.path:
    sys.path.insert(0, SEM_SEG_ROOT)

from dataset import get as get_dataset
from model import get as get_model
from utils.my_args import my_args


def seed_everything(seed=0):
    pl.seed_everything(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class CudaClearCacheCallback(pl.Callback):
    def on_validation_start(self, trainer, pl_module):
        torch.cuda.empty_cache()

    def on_validation_end(self, trainer, pl_module):
        torch.cuda.empty_cache()


def main():
    print(Fore.LIGHTRED_EX + 'PointMixer ScanNet++ checkpoint validation' + Style.RESET_ALL)

    parser = my_args()
    parser.add_argument(
        '--checkpoint_path',
        required=True,
        help='Absolute checkpoint path inside Docker.',
    )
    parser.add_argument(
        '--eval_output_dir',
        required=True,
        help='Directory where validation class metrics will be written.',
    )
    args = parser.parse_args()

    cv2.setNumThreads(0)
    seed_everything(0)
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    args.on_train = False
    args.MYCHECKPOINT = args.eval_output_dir
    os.makedirs(args.MYCHECKPOINT, exist_ok=True)

    dataset = get_dataset(args.dataset)
    val_loader = torch.utils.data.DataLoader(
        dataset.myImageFloder(args, mode=args.mode_eval),
        batch_size=args.val_batch,
        num_workers=args.val_worker,
        collate_fn=dataset.TrainValCollateFn,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    model = get_model(args.model).load_from_checkpoint(
        args.checkpoint_path,
        args=args,
        strict=args.strict_load,
    )
    if not hasattr(model, 'scheduler'):
        # validation_epoch_end logs lr through self.scheduler in the original module.
        model.configure_optimizers()

    trainer_kwargs = {
        'logger': None,
        'callbacks': [CudaClearCacheCallback()],
        'max_epochs': 1,
        'enable_checkpointing': False,
    }
    trainer_params = inspect.signature(pl.Trainer.__init__).parameters
    if 'devices' in trainer_params:
        trainer_kwargs['accelerator'] = 'gpu'
        trainer_kwargs['devices'] = args.NUM_GPUS
        if args.NUM_GPUS > 1:
            trainer_kwargs['strategy'] = (
                DDPStrategy(find_unused_parameters=False)
                if DDPStrategy is not None else 'ddp'
            )
    else:
        trainer_kwargs.pop('enable_checkpointing', None)
        trainer_kwargs['gpus'] = args.NUM_GPUS
        if args.NUM_GPUS > 1:
            if DDPPlugin is not None and 'plugins' in trainer_params:
                trainer_kwargs['plugins'] = DDPPlugin(find_unused_parameters=False)
            elif 'accelerator' in trainer_params:
                trainer_kwargs['accelerator'] = 'ddp'

    trainer = pl.Trainer(**trainer_kwargs)
    print(
        Fore.LIGHTGREEN_EX
        + 'Validate GPUS[{}], output[{}]'.format(args.NUM_GPUS, args.MYCHECKPOINT)
        + Style.RESET_ALL
    )

    trainer.validate(model, dataloaders=val_loader, verbose=True)

    metrics_dir = os.path.join(args.MYCHECKPOINT, 'class_metrics')
    print(Fore.LIGHTGREEN_EX + 'Class metrics saved to: {}'.format(metrics_dir) + Style.RESET_ALL)


if __name__ == '__main__':
    main()
