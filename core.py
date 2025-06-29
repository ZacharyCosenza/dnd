import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import logging
from typing import List, Union

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentClassifier:
    """
    A simple wrapper around a seq2seq model (e.g., FLAN-T5) to classify sentiment from prompts.
    """
    def __init__(self, path: str = None, model_name: str = "google/flan-t5-base", device: str = None):
        """
        Load tokenizer and model.

        Args:
            path (str): Optional path to a local model directory.
            model_name (str): Pretrained model name if path not provided.
            device (str): Optional device override (e.g., 'cpu', 'cuda').
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.path = path or model_name

        logger.info(f"Loading model from: {self.path} on device: {self.device}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.path)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.path).to(self.device)
        self.model.eval()

    def classify(self, prompt: str, max_tokens: int = 20) -> str:
        """
        Classify a given prompt using the model.

        Args:
            prompt (str): The input text prompt.
            max_tokens (int): Maximum tokens to generate in response.

        Returns:
            str: The model's decoded output.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=max_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class DNDDataset(Dataset):
    """
    A dataset class that splits one or more text files into sequential chunks of words.
    """
    def __init__(self, filepaths: Union[str, List[str]], max_length: int = 100, stride: int = None):
        """
        Args:
            filepaths (str or List[str]): Path(s) to .txt file(s).
            max_length (int): Max number of words per chunk.
            stride (int): Step size for sliding window. Defaults to max_length (no overlap).
        """
        if isinstance(filepaths, str):
            filepaths = [filepaths]

        self.max_length = max_length
        self.stride = stride or max_length
        self.chunks: List[str] = []

        buffer_text = ""

        for filepath in filepaths:
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    logger.info(f"Reading file: {filepath}")
                    for line in f:
                        buffer_text += line.strip() + " "
                        words = buffer_text.split()

                        # Form full chunks of max_length words
                        while len(words) >= self.max_length:
                            chunk = " ".join(words[:self.max_length])
                            self.chunks.append(chunk)
                            words = words[self.stride:]
                            buffer_text = " ".join(words)
            except FileNotFoundError:
                logger.warning(f"File not found: {filepath}")
            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")

        # Add leftover text as one final chunk
        if buffer_text.strip():
            self.chunks.append(buffer_text.strip())

        logger.info(f"Loaded {len(self.chunks)} chunks.")

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> str:
        return self.chunks[idx]
