import os

import pandas as pd
import pytest
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data.test_dataset import CLRPTestDataset


@pytest.mark.parametrize("tokenizer_name", ["bert-base-cased", "roberta-base"])
def test_clrp_test_dataset_counting_total_dataset_size(tokenizer_name):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "res", "test.csv"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = CLRPTestDataset(data, tokenizer, max_seq_len=256)
    assert len(dataset) == len(data)


@pytest.mark.parametrize("tokenizer_name", ["bert-base-cased", "roberta-base"])
def test_clrp_test_dataset_returning_id_and_tensor_dict(tokenizer_name):
    data = pd.read_csv(os.path.join(os.path.dirname(__file__), "res", "test.csv"))
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    dataset = CLRPTestDataset(data, tokenizer, max_seq_len=256)
    dataloader = DataLoader(dataset, batch_size=2)

    batch = next(iter(dataloader))
    assert all(isinstance(x, str) for x in batch[0])
    assert all(x.size(0) == 2 for x in batch[1].values())
    assert batch[1]["input_ids"].shape == (2, 256)
    assert batch[1]["attention_mask"].shape == (2, 256)
