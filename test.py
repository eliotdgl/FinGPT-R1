import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, pipeline
from FinGPTR1_pipeline.FGPTR1_tokenizer import FinGPTR1_Tokenizer

PATH = "FinGPTR1_pipeline/models/NoMLP"

from Sentiment_Analysis.sentiment_model_class import Sentiment_Analysis_Model
from Sentiment_Analysis.controlled_environment import TextDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os
from Sentiment_Analysis.data_loader import get_train_test_split
import pickle
from tqdm import tqdm






"""
# Load the CSV file
print("\nLoading dataset...\n")

with open('data/local_data/dataset_train_all_agree.pkl', 'rb') as f:
    dataset_train_all_agree = pickle.load(f)
print("\nDataset LOADED\n")
sentiment_model = Sentiment_Analysis_Model(model_name=PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train_all_agree)
print("\nModel TRAINED\n")
sentiment_model.save(base_path="Sentiment_Analysis/models/NoMLP", timestamp_name="NoMLP", keep_last=3)

print("\nModel saved to: Sentiment_Analysis/models/NoMLP\n")
"""

"""
sentiment_model = Sentiment_Analysis_Model(load_model=True)
sentiment_model.load(base_path="Sentiment_Analysis/models/NoMLP")
"""
# Load tokenizer and model
model_name = "yiyanghkust/finbert-tone"
#Next step : use our own tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
#Further step : use our own RL based model
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Create sentiment analysis pipeline WE CAN USE IT WE OUR OWN !!
nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

df = pd.read_csv('data/local_data/test_all_agree.csv', header=None, names=["Sentence", "Label"], skiprows=1)

pred_labels=[]
correct_labels = np.array(df["Label"])
print(correct_labels[:10])

with torch.no_grad():
    for i, line in tqdm(df.iterrows()):
        pred_labels.append(nlp(line["Sentence"]))
        if i>10:
            break

pred_labels = np.array(pred_labels)

print(pred_labels)
print('Result: ', np.sum(pred_labels == correct_labels))



"""
# === Load base model and tokenizer ===
model = AutoModelForSequenceClassification.from_pretrained("FinGPTR1_pipeline/models/NoMLP")
tokenizer = AutoTokenizer.from_pretrained("FinGPTR1_pipeline/models/NoMLP")


# Optional: set to eval mode
model.eval()

# Ensure model is on the right device (GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Provide a prompt (can include your special tokens if needed)
prompt = "The population is approximately <SON>"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

# Generate a completion
with torch.no_grad():
    output = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=50,
        do_sample=True,
        top_p=0.9,
        temperature=0.7,
        num_return_sequences=1
    )

# Decode the generated tokens
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
print("Generated text:\n", generated_text)
"""