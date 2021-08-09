import argparse
import os
import time
import warnings
from functools import partial

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from optuna import TrialPruned
from optuna.trial import Trial
from pytorch_lightning.loggers import WandbLogger
from transformers import logging

from lightning.callbacks import (
    BestScoreCallback,
    OptunaPruningCallback,
    ValidationEpochLogger,
)
from lightning.finetuning import CLRPFinetuningDataModule, CLRPFinetuningModule
from tuning import SearchSpace, create_study
from utils import override_from_argparse

# Disable warnings and error messages.
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Find amp backend by checking if `apex` library is installed.
try:
    import apex
except ModuleNotFoundError:
    amp_backend = "native"
else:
    amp_backend = apex.__name__


def objective(trial: Trial, search_space: SearchSpace) -> float:
    cfg = search_space(trial)
    model_name = f"{cfg.model.name.split('/')[-1]}-fold{cfg.data.fold_index}"

    pruning_callback = OptunaPruningCallback(
        trial, monitor="val/rmse", mode="min", report_best_score=True
    )
    best_score_callback = BestScoreCallback(monitor="val/rmse", mode="min")

    pl.Trainer(
        gpus=1,
        precision=16,
        amp_backend=amp_backend,
        max_epochs=cfg.train.epochs,
        accumulate_grad_batches=cfg.train.accumulate_grads,
        gradient_clip_val=cfg.optim.max_grad_norm,
        logger=WandbLogger(
            project="CLRP-Experiment",
            group=model_name,
            job_type=f"experiment-{search_space.experiment_level}",
            name=f"trial-{trial.number}",
            reinit=True,
        ),
        callbacks=[pruning_callback, best_score_callback, ValidationEpochLogger()],
        checkpoint_callback=False,
        val_check_interval=cfg.train.validation_ratio,
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    ).fit(model=CLRPFinetuningModule(cfg), datamodule=CLRPFinetuningDataModule(cfg))

    if pruning_callback.pruned:
        # If the trial is pruned, then finish the wandb logging process and raise
        # `TrialPruned` error to notify that the trial is pruned to the optuna system.
        wandb.finish()
        raise TrialPruned()
    else:
        OmegaConf.save(cfg, f"{model_name}.yaml")
        wandb.save(f"{model_name}.yaml")

        wandb.log({"best_score": best_score_callback.best_score})
        wandb.finish()

    return best_score_callback.best_score


def main(cfg: DictConfig, start_experiment: int = 0):
    search_space = SearchSpace(cfg.run.config, random_seed=int(time.time()))

    for i, experiment in enumerate(cfg.run.experiments[start_experiment:]):
        search_space.set_experiment_level(start_experiment + i)
        study, n_trials = create_study(experiment, search_space)

        study.optimize(
            partial(objective, search_space=search_space),
            n_trials=n_trials,
            gc_after_trial=True,
        )
        search_space.update_parameters(study.best_params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--start-experiment", type=int, default=0)
    args, _ = parser.parse_known_args()

    # Load pretraining configuration file and override from argparse.
    cfg = OmegaConf.load(args.config)
    override_from_argparse(cfg.run.config)

    main(cfg, args.start_experiment)
