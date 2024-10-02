from pathlib import Path
from typing import Dict, Union

import lightning.pytorch as pl
import torch
from config import Config, ModuleConfig
from datamodule import tokenize_text
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from peft import LoraConfig, TaskType, get_peft_model
from torchmetrics import MetricCollection
from torchmetrics.classification import Accuracy, F1Score, Precision, Recall
from transformers import AutoModelForSequenceClassification

lora_config = LoraConfig(
    r=ModuleConfig.lora_r,
    lora_alpha=ModuleConfig.lora_alpha,
    lora_dropout=ModuleConfig.lora_dropout,
    bias=ModuleConfig.lora_bias,
    task_type=TaskType[ModuleConfig.lora_task_type],
    target_modules=ModuleConfig.lora_target_modules,
)


class SequenceClassificationModule(pl.LightningModule):
    """
    PyTorch Lightning module for binary sequence classification using
    transformers with LoRA.
    """

    def __init__(
        self,
        model_name: str = ModuleConfig.model_name,
        learning_rate: float = ModuleConfig.learning_rate,
    ) -> None:
        """
        Initialize the SequenceClassificationModule.

        Args:
            model_name (str): Name of the pre-trained model.
            learning_rate (float): Learning rate for the optimizer.
        """
        super().__init__()

        self.model_name = model_name
        self.model = get_peft_model(AutoModelForSequenceClassification.from_pretrained(model_name), lora_config)
        self.learning_rate = learning_rate

        metrics = MetricCollection([
            Accuracy(task="binary"),
            F1Score(task="binary"),
            Precision(task="binary"),
            Recall(task="binary"),
        ])
        self.train_metrics = metrics.clone(prefix="train_")
        self.valid_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """
        Operates on a single batch of data from the training set.

        Args:
            batch (dict): A dictionary containing the input batch data.
            batch_idx (int): The index of the current batch.

        Returns:
            torch.Tensor: The computed loss for the batch.
        """
        outputs = self.model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        output = self.train_metrics(predicted_labels, batch["label"])
        self.log_dict(output, prog_bar=True, on_step=False, on_epoch=True)
        self.log("train-loss", outputs["loss"])
        return outputs["loss"]

    def validation_step(self, batch, batch_idx) -> None:
        """
        Operates on a single batch of data from the validation set.

        Args:
            batch (dict): A dictionary containing the input batch data.
            batch_idx (int): The index of the current batch.
        """
        outputs = self.model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )
        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        output = self.valid_metrics(predicted_labels, batch["label"])

        self.log_dict(output, prog_bar=True, on_epoch=True, on_step=True)
        self.log("val_loss", outputs["loss"], prog_bar=True)

    def test_step(self, batch, batch_idx) -> None:
        """
        Operates on a single batch of data from the test set.

        Args:
            batch (dict): A dictionary containing the input batch data.
            batch_idx (int): The index of the current batch.
        """
        outputs = self.model(
            batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"],
        )

        logits = outputs["logits"]
        predicted_labels = torch.argmax(logits, 1)
        output = self.test_metrics(predicted_labels, batch["label"])

        self.log_dict(output, prog_bar=True)

    def predict_step(
        self, sequence: str, cache_dir: Union[str, Path] = Config.cache_dir
    ) -> Dict[str, Union[str, float]]:
        """
        Predict the label of a given text sequence.

        Args:
            sequence (str): The input text sequence to classify.
            cache_dir (Union[str, Path]): Directory for caching the tokenizer.

        Returns:
            Dict[str, Union[str, float]]: A dictionary containing the predicted label
            and its probability.
        """
        batch = tokenize_text(sequence, model_name=self.model_name, cache_dir=cache_dir)
        # autotokenizer may cause tokens to lose device type and cause failure
        batch = batch.to(self.device)
        outputs = self.model(batch["input_ids"])
        logits = outputs["logits"]

        # Apply softmax to get probabilities
        probabilities = torch.nn.functional.softmax(logits, dim=1)

        predicted_label_id = torch.argmax(logits, dim=1).item()
        predicted_probability = probabilities[0, predicted_label_id].item()

        labels = {0: "human", 1: "ai"}
        predicted_label = labels[predicted_label_id]

        return {"label": predicted_label, "probability": predicted_probability}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """
        Configure the Adam optimizer with the specified learning rate.

        Returns:
            OptimizerLRScheduler: The configured optimizer.
        """
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
