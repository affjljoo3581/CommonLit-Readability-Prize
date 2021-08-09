from typing import Any, Dict

import optuna
import pytorch_lightning as pl
import torch
import torch.nn as nn
from optuna.distributions import BaseDistribution
from optuna.pruners import MedianPruner
from optuna.samplers import GridSampler
from optuna.study import Study
from optuna.trial import FrozenTrial
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from lightning.callbacks.pruning import OptunaPruningCallback


class SortedGridSampler(GridSampler):
    def sample_relative(
        self,
        study: Study,
        trial: FrozenTrial,
        search_space: Dict[str, BaseDistribution],
    ) -> Dict[str, Any]:
        target_grids = self._get_unvisited_grid_ids(study)
        if len(target_grids) == 0:
            target_grids = list(range(len(self._all_grids)))

        # Instead of sampling randomly, this class will register the first grid from
        # available samples.
        grid_id = target_grids[0]

        study._storage.set_trial_system_attr(
            trial._trial_id, "search_space", self._search_space
        )
        study._storage.set_trial_system_attr(trial._trial_id, "grid_id", grid_id)
        return {}


class MyLightningModule(pl.LightningModule):
    def __init__(self, coeff: float = 0.0):
        super().__init__()
        self.p = nn.Parameter(torch.zeros(()))
        self.coeff = coeff

    def training_step(self, batch, batch_idx):
        return self.p * batch[0].float().mean()

    def validation_step(self, batch, batch_idx):
        self.log("val/loss", self.trainer.current_epoch - self.coeff)

    def configure_optimizers(self):
        return SGD(self.parameters(), lr=1)


def test_optuna_pruning_callback_reporting_metrics_and_pruning_worse_trials():
    final_epochs = {}

    def objective(trial):
        coeff = trial.suggest_categorical("coeff", [0, 1, 2])

        train_dataloader = DataLoader(TensorDataset(torch.arange(1000)), batch_size=32)
        val_dataloader = DataLoader(TensorDataset(torch.arange(100)), batch_size=16)

        trainer = pl.Trainer(
            logger=None,
            checkpoint_callback=False,
            num_sanity_val_steps=0,
            max_epochs=10,
            callbacks=[
                OptunaPruningCallback(trial, "val/loss", "min", report_best_score=True)
            ],
        )
        trainer.fit(MyLightningModule(coeff), train_dataloader, val_dataloader)
        final_epochs[trial._trial_id] = trainer.current_epoch

        return 0

    optuna.create_study(
        sampler=SortedGridSampler({"coeff": [0, 1, 2]}),
        pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=5),
        direction="maximize",
    ).optimize(objective)

    assert final_epochs == {0: 9, 1: 9, 2: 5}
