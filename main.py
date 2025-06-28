from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load tokenizer and model
# tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-base")
# model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")

model = AutoModelForSeq2SeqLM.from_pretrained("models")
tokenizer = AutoTokenizer.from_pretrained("models")

# Define prompt
# prompt = "Classify sentiment: this model is amazingly bad"

# # Tokenize input
# inputs = tokenizer(prompt, return_tensors="pt")

# # Generate output
# outputs = model.generate(**inputs, max_new_tokens=20)

# # Decode and print
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# model.save_pretrained("models")
# tokenizer.save_pretrained("models")

import snscrape.modules.twitter as sntwitter
# from transformers import T5Tokenizer, T5ForConditionalGeneration
from datetime import date
# import torch

# Parameters
username = "elonmusk"  # Replace with target username
today = date.today().isoformat()

# Load model
# model_path = "flan-t5-base-classifier"  # Path to your fine-tuned model
# tokenizer = T5Tokenizer.from_pretrained(model_path)
# model = T5ForConditionalGeneration.from_pretrained(model_path)

# Scrape tweets from today
tweets = []
for tweet in sntwitter.TwitterUserScraper(username).get_items():
    if tweet.date.date().isoformat() != today:
        break
    tweets.append(tweet.content)

# Classify tweets
for text in tweets:
    input_text = f"classify: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", truncation=True)
    output_ids = model.generate(**inputs)
    label = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    print(f"TWEET: {text}\nLABEL: {label}\n{'-'*80}")
