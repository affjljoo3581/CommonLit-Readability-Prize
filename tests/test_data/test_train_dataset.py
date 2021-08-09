import os

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.train_dataset import CLRPTrainDataset


@pytest.mark.parametrize("tokenizer_name", ["bert-base-cased", "roberta-base"])
def test_clrp_train_dataset_counting_total_dataset_size(tokenizer_name):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "res", "train.csv"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = CLRPTrainDataset(data, tokenizer, max_seq_len=256)
    assert len(dataset) == len(data)


@pytest.mark.parametrize("tokenizer_name", ["bert-base-cased", "roberta-base"])
def test_clrp_train_dataset_returning_tensor_dict(tokenizer_name):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "res", "train.csv"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = CLRPTrainDataset(data, tokenizer, max_seq_len=256)
    dataloader = DataLoader(dataset, batch_size=2)

    batch = next(iter(dataloader))
    assert all(x.size(0) == 2 for x in batch.values())
    assert batch["input_ids"].shape == (2, 256)
    assert batch["attention_mask"].shape == (2, 256)
    assert batch["labels"].dtype == torch.float16
