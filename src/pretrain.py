import argparse
import os
import tarfile
import warnings

import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer, logging

from lightning.pretraining import CLRPPretrainingDataModule, CLRPPretrainingModule
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
    model_name = f"{cfg.model.name.split('/')[-1]}-clrp"

    model = CLRPPretrainingModule(cfg)
    datamodule = CLRPPretrainingDataModule(cfg)

    pl.Trainer(
        gpus=1,
        precision=16,
        amp_backend=amp_backend,
        max_epochs=cfg.train.epochs,
        gradient_clip_val=cfg.optim.max_grad_norm,
        logger=WandbLogger(
            project="CLRP-Pretraining", name=f"{cfg.model.name.split('/')[-1]}"
        ),
        checkpoint_callback=False,
        progress_bar_refresh_rate=1,
        num_sanity_val_steps=0,
        log_every_n_steps=10,
    ).fit(model, datamodule=datamodule)

    # Save the pretrained tokenizer and model weights.
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    tokenizer.save_pretrained(model_name)
    model.model.save_pretrained(model_name)

    # Compress the saved tokenizer and model weights and upload to the wandb storage.
    with tarfile.open(f"{model_name}.tar.gz", "w:gz", compresslevel=4) as tar:
        tar.add(model_name, arcname=".")
    wandb.save(f"{model_name}.tar.gz")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, _ = parser.parse_known_args()

    # Load pretraining configuration file and override from argparse.
    cfg = OmegaConf.load(args.config)
    override_from_argparse(cfg)

    main(cfg)
