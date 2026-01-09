import torch
import yaml
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
import pytorch_lightning as pl

from dataset import TranslationDataset

with open("configs/config.yaml") as f:
    cfg = yaml.safe_load(f)

class MT5FineTuner(pl.LightningModule):
    def __init__(self, model, lr):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

def main():
    df = pd.read_csv(cfg["data"]["train_path"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model"]["name"])
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["model"]["name"],
        device_map="auto",
        torch_dtype=torch.float32
    )

    lora = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=["q", "v"],
        task_type="SEQ_2_SEQ_LM"
    )

    model = get_peft_model(model, lora)

    dataset = TranslationDataset(df, tokenizer, cfg["training"]["max_length"])
    loader = DataLoader(
        dataset,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True
    )

    trainer = pl.Trainer(
        max_epochs=cfg["training"]["epochs"],
        accelerator="auto",
        devices="auto",
        precision=16
    )

    trainer.fit(MT5FineTuner(model, cfg["training"]["lr"]), loader)

    model.save_pretrained(cfg["training"]["output_dir"])
    tokenizer.save_pretrained(cfg["training"]["output_dir"])

if __name__ == "__main__":
    main()
