import argparse
from typing import Tuple

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from data import CLRPTestDataset
from modeling.models import AutoModelForCustomSequenceClassification


def prepare_model_and_dataloader(
    args: argparse.Namespace,
) -> Tuple[nn.Module, DataLoader]:
    # Load pretrained transformer model with desired dtype.
    model = AutoModelForCustomSequenceClassification.from_pretrained(
        args.model_path, num_labels=1
    )
    model.cuda().eval()

    # Create dataset and dataloader for test data.
    dataset = CLRPTestDataset(
        data=pd.read_csv(args.data_path),
        tokenizer=AutoTokenizer.from_pretrained(args.model_path),
        max_seq_len=args.max_seq_len,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    return model, dataloader


@torch.no_grad()
def main(args: argparse.Namespace):
    model, dataloader = prepare_model_and_dataloader(args)

    preds = []
    for ids, batch in dataloader:
        outputs = model(**{k: v.cuda() for k, v in batch.items()})
        logits = outputs.logits.squeeze(-1).tolist()

        for id, target in zip(ids, logits):
            preds.append({"id": id, "target": target})

    pd.DataFrame(preds).to_csv("submission.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--model_path")
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    main(args)
