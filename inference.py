from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_PATH = "checkpoints/mt5-lora-en-nl"

def format_translation(text):
    return f"translate English to Dutch: {text}"

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH)
model.eval()

def translate(sentence):
    inputs = tokenizer(format_translation(sentence), return_tensors="pt")
    outputs = model.generate(**inputs, num_beams=4, max_new_tokens=128)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    print(translate("The project deadline is approaching fast."))
