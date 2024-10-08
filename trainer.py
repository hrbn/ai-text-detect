import os
from time import perf_counter
from typing import Optional, Union

import torch

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from pytorch_lightning.loggers import WandbLogger

from lightning.pytorch.profilers import PyTorchProfiler

# Local imports
from datamodule import AutoTokenizerDataModule
from module import SequenceClassificationModule
from utils import log_perf
from config import Config, DataModuleConfig, ModuleConfig, TrainerConfig


def create_dirs(dirs: list) -> None:
    """Create directories if they do not exist"""
    for d in dirs:
        os.makedirs(d, exist_ok=True)


# Configuration variables
model_name = ModuleConfig.model_name
dataset_path = DataModuleConfig.dataset_path

# Directory paths from config
cache_dir = Config.cache_dir
log_dir = Config.log_dir
ckpt_dir = Config.ckpt_dir
prof_dir = Config.prof_dir
perf_dir = Config.perf_dir

# Create necessary directories
create_dirs([cache_dir, log_dir, ckpt_dir, prof_dir, perf_dir])

# Setting floating point precision for matrix multiplication in PyTorch
torch.set_float32_matmul_precision("medium")


def train(
    accelerator: str = TrainerConfig.accelerator,
    devices: Union[int, str] = TrainerConfig.devices,
    strategy: str = TrainerConfig.strategy,
    precision: Optional[str] = TrainerConfig.precision,
    max_epochs: int = TrainerConfig.max_epochs,
    lr: float = ModuleConfig.learning_rate,
    val_check_interval: int = TrainerConfig.val_check_interval,
    check_val_every_n_epoch: Optional[int] = TrainerConfig.check_val_every_n_epoch,
    batch_size: int = DataModuleConfig.batch_size,
    perf: bool = False,
    profile: bool = False,
) -> None:
    """
    Trains a sequence classification model using PyTorch Lightning.

    Args:
        accelerator (str): The type of accelerator to use.
        devices (Union[int, str]): The devices to use.
        strategy (str): The strategy for distributed training.
        precision (Optional[str]): Floating point precision level.
        max_epochs (int): Maximum number of training epochs.
        lr (float): Learning rate for the optimizer.
        val_check_interval (int): Interval (in steps) for validation checks during training.
        check_val_every_n_epoch (Optional[int]): Frequency (in epochs) for validation checks.
        batch_size (int): Batch size for data loading.
        perf (bool): Whether to log performance metrics.
        profile (bool): Whether to profile the training process.

    Returns:
        None
    """
    # Initialize data module
    lit_datamodule = AutoTokenizerDataModule(
        model_name=model_name,
        dataset_path=dataset_path,
        cache_dir=cache_dir,
        batch_size=batch_size,
    )

    # Initialize module
    lit_module = SequenceClassificationModule(learning_rate=lr)

    # Setup logger
    logger = WandbLogger(project="ai-text-detect", save_dir=log_dir, log_model="all")

    # Setup callbacks based on performance flag
    callbacks = [
        ModelCheckpoint(
            dirpath=ckpt_dir,
            save_top_k=1,
            monitor="val_loss",
            mode="min",
            auto_insert_metric_name=False,
            filename="best-{val_loss:.3f}",
        ),
    ]
    if not perf:
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=3))


    # Profiler setup
    profiler = PyTorchProfiler(dirpath=prof_dir) if profile else None

    # Initialize trainer
    lit_trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        precision=precision,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=callbacks,
        profiler=profiler,
        val_check_interval=val_check_interval,
        check_val_every_n_epoch=check_val_every_n_epoch,
    )

    # Start performance counter
    start = perf_counter()

    # Train the model
    lit_trainer.fit(model=lit_module, datamodule=lit_datamodule)

    # Stop performance counter
    stop = perf_counter()

    # Log performance metrics
    if perf:
        log_perf(start, stop, perf_dir, lit_trainer)


if __name__ == "__main__":
    from jsonargparse import CLI

    CLI(train, as_positional=False)
