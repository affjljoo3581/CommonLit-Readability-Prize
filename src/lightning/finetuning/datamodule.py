import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import CLRPTrainDataset


class CLRPFinetuningDataModule(pl.LightningDataModule):
    """A pytorch-lightning datamodule for CLRP-finetuning.

    This class is used for serving data from CLRP dataset to finetune the transformer
    model with sequence classification objective. Basically finetuning works with k-fold
    splits, so training and validation datasets are splitted from original training
    dataset by using k-fold. Also this class manages random seed fixing to make data
    shuffling predictable and reproducible.

    Args:
        cfg: A dictionary-based finetuning configuration object (`DictConfig`).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.train_dataset, self.val_dataset = CLRPTrainDataset.create_datasets(
            data=pd.read_csv(self.cfg.data.path),
            tokenizer=tokenizer,
            max_seq_len=self.cfg.model.max_seq_len,
            num_folds=self.cfg.data.num_folds,
            fold_index=self.cfg.data.fold_index,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=max(os.cpu_count(), 4),
            pin_memory=True,
            generator=torch.Generator().manual_seed(self.cfg.data.random_seed),
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=max(os.cpu_count(), 4),
            pin_memory=True,
        )
