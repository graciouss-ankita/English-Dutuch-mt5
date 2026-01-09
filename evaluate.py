import pandas as pd
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "checkpoints/mt5-lora-en-nl"

def format_translation(text):
    return f"translate English to Dutch: {text}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

def translate_batch(sentences):
    outputs = []
    for s in sentences:
        enc = tokenizer(format_translation(s), return_tensors="pt")
        gen = model.generate(**enc, num_beams=4, max_new_tokens=128)
        outputs.append(tokenizer.decode(gen[0], skip_special_tokens=True))
    return outputs

if __name__ == "__main__":
    df = pd.read_csv("data/test.csv")
    preds = translate_batch(df["English Source"].tolist())
    refs = df["Reference Translation"].tolist()

    print("BLEU:", sacrebleu.corpus_bleu(preds, [refs]).score)
    print("chrF++:", sacrebleu.corpus_chrf(preds, [refs]).score)
v
