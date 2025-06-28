import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from torch.utils.data import Dataset

class SentimentClassifier:
    def __init__(self, path: str = None, model_name: str = "google/flan-t5-base", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path or model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path).to(self.device)
        self.model.eval()

    def classify(self, prompt: str, max_tokens: int = 20) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

from torch.utils.data import Dataset

class DNDDataset(Dataset):
    def __init__(self, filepath, max_length=100, stride=None):
        self.max_length = max_length
        self.stride = stride or max_length
        self.chunks = []

        buffer_text = ""

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                buffer_text += line.strip() + " "
                words = buffer_text.split()
                while len(words) >= self.max_length:
                    chunk = " ".join(words[:self.max_length])
                    self.chunks.append(chunk)
                    words = words[self.stride:]
                    buffer_text = " ".join(words)

        # Add remaining text
        if buffer_text.strip():
            self.chunks.append(buffer_text.strip())

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        return self.chunks[idx]
