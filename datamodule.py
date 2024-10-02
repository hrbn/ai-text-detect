import os
from datetime import datetime
from pathlib import Path
from typing import Union

from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
import lightning.pytorch as pl
from lightning.pytorch.utilities.rank_zero import rank_zero_info
from lightning.pytorch.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from config import Config, DataModuleConfig, ModuleConfig


class AutoTokenizerDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule that downloads, caches, tokenizes, and loads the text data.
    """

    def __init__(
        self,
        dataset_path: str = DataModuleConfig.dataset_path,
        cache_dir: Union[str, Path] = Config.cache_dir,
        model_name: str = ModuleConfig.model_name,
        batch_size: int = DataModuleConfig.batch_size,
        train_size: float = DataModuleConfig.train_size,
        num_workers: int = DataModuleConfig.num_workers,
        seed: int = Config.seed,
    ) -> None:
        """
        Initialize the AutoTokenizerDataModule.

        Args:
            dataset_path (str): Path to the dataset file.
            cache_dir (Union[str, Path]): Directory for caching.
            model_name (str): Name of the pretrained model and tokenizer.
            batch_size (int): Number of samples per batch.
            train_size (float): Proportion of data to use in training.
            num_workers (int): Number of data loading worker processes.
            seed (int): Random seed for reproducibility.
        """
        super().__init__()

        self.dataset_path = dataset_path
        self.cache_dir = cache_dir
        self.model_name = model_name
        self.train_size = train_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

    def prepare_data(self) -> None:
        """
        Prepare the dataset by downloading and caching it if necessary.

        Note:
            This method should not assign state.
        """
        pl.seed_everything(self.seed)
        # Disable parallelism to avoid deadlocks
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

        if not os.path.isdir(self.cache_dir):
            os.mkdir(self.cache_dir)

        cache_dir_is_empty = len(os.listdir(self.cache_dir)) == 0

        if cache_dir_is_empty:
            rank_zero_info(f"[{datetime.now()!s}] Downloading dataset.")
            load_dataset(self.dataset_path, cache_dir=self.cache_dir)
        else:
            rank_zero_info(f"[{datetime.now()!s}] Data cache exists. Loading from cache.")

    def setup(self, stage: str) -> None:
        """
        Set up the datasets for each stage of the training process.

        Args:
            stage (str): Current stage of training ('fit', 'validate', 'test', or 'predict').
        """
        if stage == "fit" or stage is None:
            # Load and split dataset
            df = pd.read_csv(self.dataset_path)
            dataset = Dataset.from_pandas(df)
            dataset = dataset.class_encode_column("label")
            dataset = dataset.train_test_split(train_size=self.train_size, stratify_by_column="label", seed=self.seed)
            # Prepare training data
            self.train_data = dataset["train"].map(
                tokenize_text,
                batched=True,
                batch_size=None,
                fn_kwargs={"model_name": self.model_name, "cache_dir": self.cache_dir},
            )
            self.train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            # Prepare validation data
            self.val_data = dataset["test"].map(
                tokenize_text,
                batched=True,
                batch_size=None,
                fn_kwargs={"model_name": self.model_name, "cache_dir": self.cache_dir},
            )
            self.val_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
            # Free memory from unneeded dataset object
            del dataset

        if stage == "test" or stage is None:
            self.test_data = load_dataset(self.dataset_path, split="test", cache_dir=self.cache_dir)
            self.test_data.map(
                tokenize_text,
                batched=True,
                batch_size=None,
                fn_kwargs={"model_name": self.model_name, "cache_dir": self.cache_dir},
            )
            self.test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        """
        Create the training DataLoader.

        Returns:
            DataLoader: PyTorch DataLoader for the training dataset.
        """
        return DataLoader(
            self.train_data,
            num_workers=self.num_workers,
            persistent_workers=True,
            batch_size=self.batch_size,
            shuffle=True,
        )

    def val_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create the validation DataLoader.

        Returns:
            DataLoader: PyTorch DataLoader for the validation dataset.
        """
        return DataLoader(
            self.val_data,
            num_workers=self.num_workers,
            persistent_workers=True,
            batch_size=self.batch_size,
        )

    def test_dataloader(self) -> EVAL_DATALOADERS:
        """
        Create the test DataLoader.

        Returns:
            DataLoader: PyTorch DataLoader for the test dataset.
        """
        return DataLoader(
            self.test_data,
            num_workers=self.num_workers,
            persistent_workers=True,
            batch_size=self.batch_size,
        )


def tokenize_text(
    batch: dict,
    *,
    model_name: str,
    cache_dir: Union[str, Path],
    truncation: bool = True,
    padding: bool = True,
) -> dict:
    """
    Tokenize a batch of text data using a Hugging Face AutoTokenizer.

    This function can handle both single strings and dictionaries with a 'text' key.

    Args:
        batch (dict): A dictionary containing the batch data, or a string if single text input.
        model_name (str): The name of the pretrained model and tokenizer.
        cache_dir (Union[str, Path]): Path to the cache directory.
        truncation (bool, optional): Whether to truncate sequences. Default is True.
        padding (bool, optional): Whether to pad sequences. Default is True.

    Returns:
        dict: A dictionary containing tokenized inputs with 'input_ids', 'attention_mask',and 'label' fields.

    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
    text = batch if isinstance(batch, str) else batch["text"]  # Allow for inference input as raw text
    return tokenizer(text, truncation=truncation, padding=padding, return_tensors="pt")
