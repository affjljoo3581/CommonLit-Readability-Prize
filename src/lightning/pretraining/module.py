from typing import Any, Dict, List, Tuple

import torch
from omegaconf import DictConfig
from torch.optim import Optimizer
from transformers import AutoModelForMaskedLM

from lightning.extensions import ExtendedLightningModule
from optimization import LinearDecayLR

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class CLRPPretrainingModule(ExtendedLightningModule):
    """A pytorch-lightning module for CLRP-pretraining.

    This class contains entire training and validation procedures with configuring an
    optimizer and its learning rate scheduler. To pretrain the transformer model with
    CLRP dataset, `*ForMaskedLM` is used to train with masked-lm objective. Contrary to
    the finetuning, the optimizer does not ignore `LayerNorm` parameters and biases from
    weight-decaying.

    Args:
        cfg: A dictionary-based pretraining configuration object (`DictConfig`).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.model = AutoModelForMaskedLM.from_pretrained(cfg.model.name)

        self.save_hyperparameters(cfg)

    def training_step(self, batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        loss = self.model(**batch).loss
        self.log("train/loss", loss)
        return loss

    def validation_step(self, batch: Dict[str, torch.Tensor], idx: int):
        self.log("val/loss", self.model(**batch).loss)

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.optim.learning_rate,
            weight_decay=self.cfg.optim.weight_decay,
            betas=(0.9, 0.98),
            eps=1e-6,
        )
        scheduler = LinearDecayLR(
            optimizer,
            total_steps=self.num_training_steps,
            warmup_steps=int(self.num_training_steps * self.cfg.train.warmup_ratio),
        )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]
