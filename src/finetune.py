import argparse
import os
import tarfile
import warnings

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, logging

from lightning.callbacks import ValidationEpochLogger
from lightning.finetuning import CLRPFinetuningDataModule, CLRPFinetuningModule
from utils import override_from_argparse

# Disable warnings and error messages.
logging.set_verbosity_error()
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Find amp backend by checking if `apex` library is installed.
try:
    import apex

    amp_backend = apex.__name__
except ModuleNotFoundError:
    amp_backend = "native"


def main(cfg: DictConfig):
    model_name = f"{cfg.model.name.split('/')[-1]}-fold{cfg.data.fold_index}"
    model_checkpoint = ModelCheckpoint(monitor="val/rmse", save_weights_only=True)

    pl.Trainer(
        gpus=1,
        precision=16,
        amp_backend=amp_backend,
        max_epochs=cfg.train.epochs,
        accumulate_grad_batches=cfg.train.accumulate_grads,
        gradient_clip_val=cfg.optim.max_grad_norm,
        logger=WandbLogger(
            project="CLRP-Finetuning",
            group=cfg.model.name.split("/")[-1],
            name=f"{cfg.model.name.split('/')[-1]}-fold{cfg.data.fold_index}",
        ),
        callbacks=[model_checkpoint, ValidationEpochLogger()],
        val_check_interval=cfg.train.validation_ratio,
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    ).fit(CLRPFinetuningModule(cfg), datamodule=CLRPFinetuningDataModule(cfg))

    wandb.log({"best_score": model_checkpoint.best_model_score})

    # Save the finetuned tokenizer and model weights.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    model = CLRPFinetuningModule.load_from_checkpoint(model_checkpoint.best_model_path)

    tokenizer.save_pretrained(model_name)
    model.model.half().save_pretrained(model_name)

    # Compress the saved tokenizer and model weights and upload to the wandb storage.
    with tarfile.open(f"{model_name}.tar.gz", "w:gz", compresslevel=4) as tar:
        tar.add(model_name, arcname=".")
    wandb.save(f"{model_name}.tar.gz")

    # Remove the checkpoint file to reduce the total file size.
    os.remove(model_checkpoint.best_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, _ = parser.parse_known_args()

    # Load pretraining configuration file and override from argparse.
    cfg = OmegaConf.load(args.config)
    override_from_argparse(cfg)

    main(cfg)
