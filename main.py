from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import time
import torch

def load_model_and_tokenizer(path: str = None, model_name: str = "google/flan-t5-base"):
    path = path or model_name
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    return tokenizer, model

def classify_sentiment(prompt: str, tokenizer, model, max_tokens: int = 20):
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    start_time = time.time()

    tokenizer, model = load_model_and_tokenizer(path="models")

    prompt = "Classify sentiment: this model is amazingly bad"
    result = classify_sentiment(prompt, tokenizer, model)

    print("Output:", result)
    print("Elapsed time: {:.2f}s".format(time.time() - start_time))

if __name__ == "__main__":
    main()
