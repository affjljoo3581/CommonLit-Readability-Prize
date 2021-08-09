from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD, Optimizer
from torch.utils.data import DataLoader, TensorDataset

from lightning.extensions import ExtendedLightningModule


def test_extended_lightning_module_calculating_steps():
    class MyLightningModule(ExtendedLightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(1, 1)

        def setup(self, stage: Optional[str] = None):
            self.dataset = TensorDataset(torch.arange(100))

        def train_dataloader(self) -> DataLoader:
            return DataLoader(self.dataset)

        def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
            return torch.zeros([], requires_grad=True)

        def configure_optimizers(self) -> Optimizer:
            assert self.num_training_steps == 1000
            return SGD(self.parameters(), lr=1)

    pl.Trainer(
        max_epochs=10,
        logger=False,
        checkpoint_callback=False,
    ).fit(MyLightningModule())


def test_extended_lightning_module_calculating_steps_with_datamodule():
    class MyLightningDataModule(pl.LightningDataModule):
        def setup(self, stage: Optional[str] = None):
            self.dataset = TensorDataset(torch.arange(100))

        def train_dataloader(self) -> DataLoader:
            return DataLoader(self.dataset)

    class MyLightningModule(ExtendedLightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(1, 1)

        def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
            return torch.zeros([], requires_grad=True)

        def configure_optimizers(self) -> Optimizer:
            assert self.num_training_steps == 1000
            return SGD(self.parameters(), lr=1)

    pl.Trainer(
        max_epochs=10,
        logger=False,
        checkpoint_callback=False,
    ).fit(MyLightningModule(), datamodule=MyLightningDataModule())


def test_extended_lightning_module_calculating_steps_with_datamodule_and_accumulation():
    class MyLightningDataModule(pl.LightningDataModule):
        def setup(self, stage: Optional[str] = None):
            self.dataset = TensorDataset(torch.arange(100))

        def train_dataloader(self) -> DataLoader:
            return DataLoader(self.dataset)

    class MyLightningModule(ExtendedLightningModule):
        def __init__(self):
            super().__init__()
            self.model = nn.Linear(1, 1)

        def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
            return torch.zeros([], requires_grad=True)

        def configure_optimizers(self) -> Optimizer:
            assert self.num_training_steps == 500
            return SGD(self.parameters(), lr=1)

    pl.Trainer(
        max_epochs=10,
        accumulate_grad_batches=2,
        logger=False,
        checkpoint_callback=False,
    ).fit(MyLightningModule(), datamodule=MyLightningDataModule())
