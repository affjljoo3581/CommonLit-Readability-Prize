from typing import Any, Dict, List, Tuple

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from transformers import AutoConfig

from lightning.extensions import ExtendedLightningModule
from modeling.models import AutoModelForCustomSequenceClassification
from optimization import LinearDecayLR, create_param_groups

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class CLRPFinetuningModule(ExtendedLightningModule):
    """A pytorch-lightning module for CLRP-finetuning.

    This class contains entire training and validation procedures with configuring an
    optimizer and its learning rate scheduler. This class uses
    `*ForCustomSequenceClassification` class, which is designed to upgrade the original
    classification model.

    This class supports separated weight decaying (i.e. do not apply to `LayerNorm`
    parameters and biases), layer-wise learning rate decaying, re-initializing a portion
    of transformer layers, etc. Since the random seeds for initializing weights are
    important to the performance, it manages random seed fixing to make weight
    initializations predictable and reproducible.

    Because the target metric of CLRP task is **RMSE**, this class gathers all batched
    validation losses (**MSE**) and apply square root to the globally-meaned mse loss to
    compute correct validation metric.

    Args:
        cfg: A dictionary-based finetuning configuration object (`DictConfig`).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.model = AutoModelForCustomSequenceClassification.from_pretrained(
            cfg.model.name,
            config=AutoConfig.from_pretrained(
                cfg.model.name,
                dropout=0.0,
                dropout_rate=0.0,
                activation_dropout=0.0,
                hidden_dropout=0.0,
                hidden_dropout_prob=0.0,
                embd_pdrop=0.0,
                resid_pdrop=0.0,
                num_labels=1,
            ),
        )
        self.model.set_classifier_dropout(cfg.model.classifier_dropout)

        # Initialize the weights with fixing the random seed.
        torch.manual_seed(cfg.model.random_seed)
        self.model.init_classifier()
        self.model.init_transformer_layers(cfg.model.num_reinit_layers, reverse=True)

        self.save_hyperparameters(cfg)

    def training_step(self, batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        loss = self.model(**batch).loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        # For the precise loss calculation, we need to scale with current batch size to
        # consider the truncated batches.
        batch_size = batch["input_ids"].size(0)
        return self.model(**batch).loss * batch_size

    def validation_epoch_end(self, outputs: List[torch.Tensor]):
        # Sometimes trainer calls one more validation epoch when the training is stopped
        # by setting `should_stop` to `True`. Because of that, the same validation
        # metrics are logged. To prevent this, the loss calculation and log writting
        # would not be performed when `should_stop` is set to `True`.
        if self.trainer.should_stop:
            return

        loss = torch.stack(outputs).sum() / len(self.trainer.datamodule.val_dataset)
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/rmse", loss.sqrt(), prog_bar=True)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = AdamW(
            create_param_groups(
                self.model,
                self.cfg.optim.learning_rate,
                self.cfg.optim.layerwise_lr_decay,
                self.cfg.optim.weight_decay,
            ),
            betas=(0.9, 0.98),
            eps=1e-6,
        )
        scheduler = LinearDecayLR(
            optimizer,
            total_steps=self.num_training_steps,
            warmup_steps=int(self.num_training_steps * self.cfg.train.warmup_ratio),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
