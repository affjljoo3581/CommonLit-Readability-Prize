from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd
import torch
from sklearn.model_selection import KFold
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase


class CLRPTrainDataset(Dataset):
    """A train dataset for **CommonLit Readability Prize** task.

    This class extracts the target excerpts (strings) and the answer labels from the
    **CommonLit Readability Prize** dataset and automatically tokenizes the sentences
    into subwords. Since this dataset class is considered to use for transformer
    sequence classification, it returns not only the outputs (encoded tensors) from the
    tokenize but also the answer label tensor. Hence, you can directly pass the outputs
    to the transformer model. See the below example.

    Examples::

        >>> dataset = CLRPTrainDataset(data, tokenizer, max_seq_len=320)
        >>> dataloader = DataLoader(dataset, ...)
        >>> for batch in dataloader:
        >>>     output = sequence_classifier(**batch)
        >>>     output.loss.backward()

    Args:
        data: The pandas dataframe which contains data for training.
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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        labels = torch.tensor(self.data.iloc[idx].target, dtype=torch.float16)
        encodings = {k: v[idx] for k, v in self.tokenized_excerpts.items()}
        return {"labels": labels, **encodings}

    @staticmethod
    def create_datasets(
        data: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_seq_len: int = 256,
        num_folds: int = 4,
        fold_index: int = 0,
    ) -> Tuple[CLRPTrainDataset, CLRPTrainDataset]:
        """
        Create train and validation datasets by splitting the original data into
        K-Fold subsets.

        Args:
            data: The pandas dataframe which contains data for training.
            tokenizer: The tokenizer of each transformer. It is used for tokenizing the
                excerpts in the dataset.
            max_seq_len: The maximum sequence length. The shorter sentences will be
                padded and the longer sentences will be truncated. Default is `256`.
            num_folds: The number of k-folds. Default is `4`.
            fold_index: The current k-fold index. Default is `0`.

        Returns:
            - A `CLRPTrainDataset` dataset with the training samples.
            - A `CLRPTrainDataset` dataset with the validation samples.
        """
        kfold = KFold(num_folds, shuffle=True, random_state=0)
        train_indices, val_indices = list(kfold.split(data))[fold_index]

        return (
            CLRPTrainDataset(data.iloc[train_indices], tokenizer, max_seq_len),
            CLRPTrainDataset(data.iloc[val_indices], tokenizer, max_seq_len),
        )
