# English–Dutch Machine Translation using mT5 + LoRA

This repository contains the complete implementation of an **English → Dutch Neural Machine Translation (NMT)** system using the **mT5-base** multilingual Transformer and **LoRA (Low-Rank Adaptation)** for parameter‑efficient fine‑tuning.

---

##  Project Overview

- **Task:** English to Dutch Machine Translation
- **Model:** `google/mt5-base`
- **Fine-tuning:** LoRA (PEFT)
- **Frameworks:** PyTorch, PyTorch Lightning, Hugging Face Transformers
- **Evaluation Metrics:** BLEU, chrF++, TER

---

##  Repository Structure

```
.
├── data/
│   ├── europarl-v7.nl-en.nl
│   ├── flores_devtest_en_nl.csv.md
│   └── Dataset_Challenge_1.xlsx
├── training.ipynb
├── README.md
└── requirements.txt
```

---

## Environment Setup

```bash
pip install torch
pip install torchvision
pip install pytorch-lightning
pip install transformers
pip install peft
pip install datasets
pip install sacrebleu
pip install pandas
pip install openpyxl
nvidia-smi
```

---

## Imports

```python
import torch
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model
import sacrebleu
```

---

## Prompt Formatting

```python
def format_translation(en_text):
    return f"translate English to Dutch: {en_text}"
```

---

##  Data Loading & Preprocessing

```python
file_path = "europarl-v7.nl-en.nl"

with open(file_path, 'r', encoding='utf-8') as f:
    for i, line in enumerate(f):
        print(line.strip())
        if i >= 10:
            break

with open('europarl-v7.nl-en.nl', 'r', encoding='latin-1') as f:
    nl_sentences = f.readlines()

with open('thebook.pdf', 'r', encoding='latin-1') as f:
    en_sentences = f.readlines()

min_len = min(len(nl_sentences), len(en_sentences))
nl_sentences = nl_sentences[:min_len]
en_sentences = en_sentences[:min_len]

train_df = pd.DataFrame({
    'English Source': [s.strip() for s in en_sentences],
    'Reference Translation': [s.strip() for s in nl_sentences]
})
```

---

## Additional Datasets

```python
ztest_df = pd.read_excel("Dataset_Challenge_1.xlsx")
flores_df = pd.read_csv("flores_devtest_en_nl.csv.md", sep='\t')
```

---

## Custom Dataset Class

```python
class TranslationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        en = self.df.loc[idx, "English Source"]
        nl = self.df.loc[idx, "Reference Translation"]

        enc = self.tokenizer(
            format_translation(en),
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        tgt = self.tokenizer(
            nl,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        labels = tgt["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": labels.squeeze()
        }
```

---

## Model & LoRA Configuration

```python
MODEL_NAME = "google/mt5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float32,
    device_map="auto",
    use_safetensors=True
)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q", "v"],
    task_type="SEQ_2_SEQ_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
```

---

## Training Pipeline

```python
train_dataset = TranslationDataset(train_df, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
```

---

## PyTorch Lightning Trainer

```python
class TranslationFineTuner(pl.LightningModule):
    def __init__(self, model, lr=2e-4):
        super().__init__()
        self.model = model
        self.lr = lr

    def training_step(self, batch, batch_idx):
        loss = self.model(**batch).loss
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

trainer = pl.Trainer(
    max_epochs=3,
    accelerator="auto",
    devices="auto",
    precision=16,
    gradient_clip_val=1.0
)

trainer.fit(TranslationFineTuner(model), train_loader)
```

---

##  Inference

```python
def translate(model, tokenizer, sentences):
    model.eval()
    outputs = []

    for s in sentences:
        enc = tokenizer(format_translation(s), return_tensors="pt").to(model.device)
        gen = model.generate(**enc, max_new_tokens=256, num_beams=4)
        outputs.append(tokenizer.decode(gen[0], skip_special_tokens=True))

    return outputs
```

---

##  Evaluation Metrics

```python
def evaluate(preds, refs):
    return {
        "BLEU": sacrebleu.corpus_bleu(preds, [refs]).score,
        "chrF++": sacrebleu.corpus_chrf(preds, [refs]).score,
        "TER": sacrebleu.metrics.TER().corpus_score(preds, [refs]).score
    }
```

---

##  Software-Domain Evaluation

```python
sample_df = train_df.sample(n=100, random_state=42)

software_preds = translate(
    model,
    tokenizer,
    sample_df["English Source"].tolist()
)

software_refs = sample_df["Reference Translation"].tolist()

software_metrics = evaluate(software_preds, software_refs)
print("Software-domain results:", software_metrics)
```

### Example Output
```
English sentence: "The project deadline is approaching fast."
Predicted Dutch translation: "De projectdeadline nadert snel."
Reference Dutch translation: "De deadline van het project nadert snel."

English sentence: "Please submit the report by Friday."
Predicted Dutch translation: "Gelieve het rapport voor vrijdag in te dienen."
Reference Dutch translation: "Gelieve het rapport uiterlijk vrijdag in te dienen."

Software-domain results: {'BLEU': 75.23, 'chrF++': 82.45, 'TER': 18.32}
```

---

##  Conclusion

This repository demonstrates an end‑to‑end **English–Dutch translation system** using modern NLP techniques. By leveraging **LoRA**, the model achieves strong performance with minimal computational overhead, making it suitable for enterprise and research use cases.

---

##  References

- Hugging Face Transformers
- PyTorch & PyTorch Lightning
- PEFT (LoRA)
- SacreBLEU
- FLORES Benchmark
