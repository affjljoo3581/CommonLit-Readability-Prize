import pytest
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from lightning.callbacks.best_score import BestScoreCallback


class MyLightningModule(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(()))

    def training_step(self, batch, batch_idx):
        return self.p * batch[0].float().mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self.trainer.current_epoch)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1)


@pytest.mark.parametrize("val_check_interval", [0.1, 0.25, 0.5, 1.0])
@pytest.mark.parametrize("mode", ["min", "max"])
def test_best_score_callback_recording_best_score(val_check_interval, mode):
    best_score_callback = BestScoreCallback("val/loss", mode)

    train_dataloader = DataLoader(TensorDataset(torch.arange(1000)), batch_size=32)
    val_dataloader = DataLoader(TensorDataset(torch.arange(100)), batch_size=16)

    pl.Trainer(
        logger=None,
        checkpoint_callback=False,
        val_check_interval=val_check_interval,
        num_sanity_val_steps=0,
        max_epochs=10,
        callbacks=[best_score_callback],
    ).fit(MyLightningModule(), train_dataloader, val_dataloader)

    if mode == "min":
        assert best_score_callback.best_score == 0
    elif mode == "max":
        assert best_score_callback.best_score == 9
