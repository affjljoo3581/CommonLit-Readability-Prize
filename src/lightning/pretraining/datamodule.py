import os
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from data import CLRPTrainDataset


class CLRPPretrainingDataModule(pl.LightningDataModule):
    """A pytorch-lightning datamodule for CLRP-pretraining.

    This class is used for serving data from CLRP dataset to pretrain the transformer
    model with masked-lm task. Note that both train and validation dataset use full
    original train dataset in CLRP.

    Args:
        cfg: A dictionary-based pretraining configuration object (`DictConfig`).
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: Optional[str] = None):
        data = pd.read_csv(self.cfg.data.path)
        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.name)

        self.dataset = CLRPTrainDataset(data, tokenizer, self.cfg.model.max_seq_len)
        self.collator = DataCollatorForLanguageModeling(tokenizer)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=max(os.cpu_count(), 4),
            collate_fn=self.collator,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.dataset,
            batch_size=self.cfg.train.batch_size,
            num_workers=max(os.cpu_count(), 4),
            collate_fn=self.collator,
            pin_memory=True,
        )
