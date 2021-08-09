from collections import defaultdict

import numpy as np
import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from lightning.callbacks.validation_epoch import ValidationEpochLogger


class MyLogger(pl.loggers.LightningLoggerBase):
    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(dict)

    @property
    def name(self):
        return "MyLogger"

    @property
    def experiment(self):
        ...

    @property
    def version(self):
        return "0.1"

    def log_hyperparams(self, params):
        ...

    def log_metrics(self, metrics, step):
        self.metrics[step].update(metrics)


class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(()))

    def training_step(self, batch, batch_idx):
        return self.p * batch[0].float().mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self.p * batch[0].float().mean())

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1)


@pytest.mark.parametrize("val_check_interval", [0.1, 0.25, 0.5])
@pytest.mark.parametrize("metric_name", ["val_epoch", "vepoch", "val/epoch"])
def test_validation_epoch_logger_logging_epochs_with_various_names(
    val_check_interval, metric_name
):
    logger = MyLogger()

    train_dataloader = DataLoader(TensorDataset(torch.arange(1000)), batch_size=32)
    val_dataloader = DataLoader(TensorDataset(torch.arange(100)), batch_size=16)

    pl.Trainer(
        logger=logger,
        checkpoint_callback=False,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=0,
        max_epochs=10,
        callbacks=[ValidationEpochLogger(metric_name)],
    ).fit(MyLightningModule(), train_dataloader, val_dataloader)

    epochs = [m[metric_name] for m in logger.metrics.values()]
    assert (np.array(epochs) == np.arange(0, 10, val_check_interval)).all()


@pytest.mark.parametrize("batch_size", [4, 8, 16, 32, 64])
def test_validation_epoch_logger_logging_epochs_with_various_batch_size(batch_size):
    logger = MyLogger()

    train_dataloader = DataLoader(
        TensorDataset(torch.arange(1000)), batch_size=batch_size
    )
    val_dataloader = DataLoader(TensorDataset(torch.arange(100)), batch_size=16)

    pl.Trainer(
        logger=logger,
        checkpoint_callback=False,
        val_check_interval=0.25,
        num_sanity_val_steps=0,
        max_epochs=10,
        callbacks=[ValidationEpochLogger("val/epoch")],
    ).fit(MyLightningModule(), train_dataloader, val_dataloader)

    epochs = [m["val/epoch"] for m in logger.metrics.values()]
    assert 0 <= min(epochs) and max(epochs) < 10
