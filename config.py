import os
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Optional, Union

this_file = Path(__file__)

this_workspace = os.getcwd()

cache_dir: str = os.path.join(this_workspace, "data")
log_dir: str = os.path.join(this_workspace, "logs")
ckpt_dir: str = os.path.join(this_workspace, "checkpoints")
prof_dir: str = os.path.join(this_workspace, "logs", "profiler")
perf_dir: str = os.path.join(this_workspace, "logs", "perf")


@dataclass
class Config:
    cache_dir: str = cache_dir
    log_dir: str = log_dir
    ckpt_dir: str = ckpt_dir
    prof_dir: str = prof_dir
    perf_dir: str = perf_dir
    seed: int = 9


@dataclass
class ModuleConfig:
    model_name: str = "distilbert/distilbert-base-cased"
    learning_rate: float = 5e-05
    lora_r: int = 4
    lora_alpha: int = 32
    lora_dropout: float = 0.01
    lora_bias: str = "none"
    lora_task_type: str = "SEQ_CLS"
    lora_target_modules: ClassVar[list[str]] = ["q_lin", "k_lin", "v_lin"]


@dataclass
class DataModuleConfig:
    dataset_path: str = "data/sample_100k.csv"
    batch_size: int = 12
    train_size: float = 0.975
    num_workers: int = 4


@dataclass
class TrainerConfig:
    accelerator: str = "auto"
    devices: Union[int, str] = "auto"
    strategy: str = "auto"
    precision: Optional[str] = "16-mixed"
    max_epochs: int = 1
    val_check_interval: int = 500
    check_val_every_n_epoch: Optional[int] = None
