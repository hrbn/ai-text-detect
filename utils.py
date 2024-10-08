import json
import os
from pathlib import Path
from typing import Union, Optional
import re

from lightning.pytorch import Trainer


def log_perf(
    start: float,
    stop: float,
    perf_dir: Union[str, Path],
    trainer: Trainer,
) -> None:
    """logs performance stats from training runs"""

    # sync to last checkpoint
    mc = [i for i in trainer.callbacks if i.__class__.__name__ == "ModelCheckpoint"]
    if mc:
        mc = mc[0]
        version = mc._last_checkpoint_saved.split("/")[-1].split(".")[0]
    else:  # this should never be triggered since the example forces use of ModelCheckpoint
        perfs = os.listdir(perf_dir)
        version = f"version_{len(perfs)}"

    perf = {
        "perf": {
            # "device_name": torch.cuda.get_device_name(),
            "num_node": trainer.num_nodes,
            "num_devices": trainer.num_devices,
            "strategy": trainer.strategy.__class__.__name__,
            "precision": trainer.precision,
            "epochs": trainer.current_epoch,
            "global_step": trainer.global_step,
            "max_epochs": trainer.max_epochs,
            "min_epochs": trainer.min_epochs,
            "batch_size": trainer.datamodule.batch_size,
            "runtime_min": (stop - start) / 60,
        }
    }

    if not os.path.isdir(perf_dir):
        os.mkdir(perf_dir)

    with open(os.path.join(perf_dir, version + ".json"), "w") as perf_file:
        json.dump(perf, perf_file, indent=4)


def best_checkpoint(directory: str = "./checkpoints") -> Optional[str]:
    pattern = re.compile(r"best-(\d+\.\d+)\.ckpt")

    best = min(
        (f for f in Path(directory).iterdir() if (match := pattern.match(f.name))),
        key=lambda f: float(pattern.match(f.name).group(1)),
        default=None,
    )
    return str(best) if best else None
