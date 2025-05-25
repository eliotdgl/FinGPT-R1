import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from datasets import load_dataset

from Sentiment_Analysis.sentiment_model_class import Sentiment_Analysis_Model
from Sentiment_Analysis.controlled_environment import TextDataGenerator
from Sentiment_Analysis.data_loader import get_train_test_split

from tokenization.preprocess_text import preprocess_text


# ===== Training =====

#INPUT_PATH = "FinGPTR1_pipeline/models/NoMLP"
OUTPUT_PATH = "models/FinBertLoRA"  #"models/NoMLPGen"

# Load the CSV file
print("\nLoading dataset...\n")

with open('data/local_data/generated_data.pkl', 'rb') as f:
    dataset_train = pickle.load(f)
print("\nDataset LOADED\n")
sentiment_model = Sentiment_Analysis_Model()
print("\nModel LOADED\n")
sentiment_model.train(dataset_train)
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")

# ====================



# ===== Inference =====
"""
sentiment_model = Sentiment_Analysis_Model(load_model=True)
sentiment_model.load(base_path=OUTPUT_PATH)

dataset = pd.read_csv('data/local_data/test_all_agree.csv', header=None, names=["Sentence", "Label"], skiprows=1)

pred_labels = []
correct_labels = []

with torch.no_grad():
    for i, line in tqdm(dataset.iterrows()):
        processed_text, _ = preprocess_text(line["Sentence"])
        pred_labels.append(sentiment_model.predict(processed_text)[1])
        correct_labels.append(line["Label"])

correct_labels = np.array(correct_labels)
pred_labels = np.array(pred_labels)

print('Result: ', np.sum(pred_labels == correct_labels))

# ====================
"""