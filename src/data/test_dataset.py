from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class CLRPTestDataset(Dataset):
    """A test dataset for **CommonLit Readability Prize** task.

    Contrary to the case of training, it only needs the excerpt texts to predict the
    readability score. So this class returns the exceprt id and its encoded output
    tensors from a tokenizer. Note that you can pass the encoded outputs to the
    transformer model directly. See the below example.

    Examples::

        >>> dataset = CLRPTestDataset(data, tokenizer, max_seq_len=320)
        >>> dataloader = DataLoader(dataset, ...)
        >>> for batch in dataloader:
        >>>     ids, encodings = batch
        >>>     logits = sequence_classifier(**encodings).logits

    Args:
        data: The pandas dataframe which contains test data.
        tokenizer: The tokenizer to encode the strings to subword tokens. It will
            tokenize the excerpts in the dataset.
        max_seq_len: The maximum sequence length. The shorter sentences will be padded
            and the longer sentences will be truncated. Default is `256`.
    """

    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 256,
    ):
        super().__init__()
        self.data = data
        self.tokenized_excerpts = tokenizer(
            data.excerpt.tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[str, Dict[str, torch.Tensor]]:
        encodings = {k: v[idx] for k, v in self.tokenized_excerpts.items()}
        return self.data.iloc[idx].id, encodings
