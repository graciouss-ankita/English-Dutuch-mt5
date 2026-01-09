# Pseudocode for English→Dutch mT5 + LoRA pipeline

This file contains concise, runnable-style pseudocode for the full pipeline described in the repository's README: data loading, tokenization, LoRA (PEFT) wrapping, training, inference and evaluation. Use it as a quick reference or to paste into a script/notebook.

```python
# HIGH LEVEL
# 1. Load data -> DataFrame with columns: "English Source", "Reference Translation"
df = load_dataset(path)

# 2. Prepare tokenizer and model (mT5) and wrap with LoRA (PEFT)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
lora_config = LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, target_modules=["q","v"], task_type="SEQ_2_SEQ_LM")
model = get_peft_model(base_model, lora_config)
# Only small adapter params are trainable

# 3. Create Dataset and DataLoader
dataset = TranslationDataset(df, tokenizer, max_length=MAX_LEN)
dataloader = DataLoader(dataset, batch_size=BATCH, shuffle=True, collate_fn=collate_fn)

# 4. Train using Lightning or plain loop
trainer = Trainer(...)
trainer.fit(lightning_module, dataloader)

# 5. Inference on texts
preds = translate(model, tokenizer, sentences)

# 6. Evaluate
metrics = evaluate(preds, refs)
```

```python
# TRAINING LOOP (DETAILED)
# shapes: input_ids (B, S_in), attention_mask (B, S_in), labels (B, S_out)
for epoch in range(num_epochs):
    for batch in dataloader:
        # Move to device (GPU/CPU)
        batch = {k: v.to(device) for k, v in batch.items()}

        # Forward: hf Seq2SeqLM accepts labels and computes loss internally
        outputs = model(
            input_ids = batch["input_ids"],         # (B, S_in)
            attention_mask = batch["attention_mask"], # (B, S_in)
            labels = batch["labels"]                # (B, S_out), PAD tokens -> -100
        )
        loss = outputs.loss  # scalar

        # Backward & update (Lightning or manual)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Optional logging / checkpointing
        log("train_loss", loss.item())
```

```python
# LABEL MASKING
# After tokenizing target (nl): tgt["input_ids"] shape (1, S)
labels = tgt["input_ids"].clone().squeeze(0)   # -> (S,)
# Replace pad token id with -100 so HF loss ignores pad positions
labels[labels == tokenizer.pad_token_id] = -100
```

```python
# SINGLE STEP SMOKE TEST (use small toy example, CPU OK)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = get_peft_model(AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME), lora_config)

# toy example
src = "translate English to Dutch: Hello, how are you?"
tgt = "Hallo, hoe gaat het met je?"

enc = tokenizer(src, return_tensors="pt", padding=True, truncation=True, max_length=64)
tgt_tokens = tokenizer(tgt, return_tensors="pt", padding=True, truncation=True, max_length=64)["input_ids"]
labels = tgt_tokens.clone()
labels[labels == tokenizer.pad_token_id] = -100

batch = {
    "input_ids": enc["input_ids"],           # (1, S_in)
    "attention_mask": enc["attention_mask"], # (1, S_in)
    "labels": labels                         # (1, S_out)
}

# one forward + backward step
outputs = model(**batch)
loss = outputs.loss
loss.backward()
# check gradients exist only for LoRA parameters (trainable)
for name, p in model.named_parameters():
    if p.requires_grad:
        print("trainable:", name)
```

```python
# GENERATION
model.eval()
for sentence in sentences:
    enc = tokenizer("translate English to Dutch: " + sentence, return_tensors="pt").to(model.device)
    gen_ids = model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, num_beams=NUM_BEAMS)
    pred = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    outputs.append(pred)
```

```python
# EVALUATION (sacrebleu expects list-of-hypotheses and list-of-list-of-refs)
bleu = sacrebleu.corpus_bleu(hypotheses, [references]).score
chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
ter  = sacrebleu.metrics.TER().corpus_score(hypotheses, [references]).score
```

## Key notes

- Set tokenizer.pad_token if missing (T5 family often lacks it).
- Mask labels with -100 so padding doesn't contribute to loss.
- Use get_peft_model to wrap the base model with LoRA; only adapters will be trainable.
- For large models avoid huge max_length or batches to prevent OOM; prefer dynamic padding.
- When using device_map or mixed device setups, ensure compatibility with Lightning/PEFT.


End of file
