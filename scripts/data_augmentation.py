import argparse
from typing import Tuple

import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
)


class MyDataset(Dataset):
    def __init__(self, data: pd.DataFrame):
        super().__init__()
        self.data = data

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple[str, str]:
        return self.data.iloc[index].id, self.data.iloc[index].excerpt


@torch.no_grad()
def main(args: argparse.Namespace):
    # Create tokenizer, model and data-collator.
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name).cuda().eval()

    collator = DataCollatorForLanguageModeling(
        tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    # Load data and create dataset, dataloader.
    data = pd.read_csv(args.data_path)
    dataset = MyDataset(data)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True)

    # Generate augmented texts using masked-lm.
    generated_data = []
    for ids, excerpts in dataloader:
        # Tokenize the excerpts and apply token masking.
        batch_inputs = tokenizer(excerpts, padding="longest", return_tensors="pt")

        input_ids, mask_labels = collator.mask_tokens(batch_inputs["input_ids"])
        batch_inputs["input_ids"] = input_ids

        # Predict the masked tokens and decode to the texts.
        output = model(**{k: v.cuda() for k, v in batch_inputs.items()})
        output_ids = output.logits.argmax(dim=-1).cpu()

        generated_ids = torch.where(mask_labels == -100, input_ids, output_ids)
        generated = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        for id, excerpt in zip(ids, generated):
            generated_data.append({"id": id, "excerpt": excerpt})

    # Save the generated dataframe.
    pd.DataFrame(generated_data).to_csv("generated.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    parser.add_argument("--model_name", default="roberta-large")
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--mlm_probability", default=0.25, type=float)
    args = parser.parse_args()

    main(args)
