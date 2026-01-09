import torch
from torch.utils.data import Dataset

def format_translation(text):
    return f"translate English to Dutch: {text}"

class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        src = self.df.loc[idx, "English Source"]
        tgt = self.df.loc[idx, "Reference Translation"]

        enc = self.tokenizer(
            format_translation(src),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        dec = self.tokenizer(
            tgt,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = dec["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
