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


# Load the CSV file
print("\nLoading dataset...\n")

with open('data/local_data/generated_data.pkl', 'rb') as f:
    dataset_train = pickle.load(f)
print("\nDataset LOADED\n")


# ===== Training =====

INPUT_PATH = "FinGPTR1_pipeline/models/NoMLP"
OUTPUT_PATH = "models/NoMLPLoRA"

sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")

# ====================

INPUT_PATH = "FinGPTR1_pipeline/models/NoMLP"
OUTPUT_PATH = "models/NoMLPLoRAWhole"

sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_', 'embeddings', 'classifier'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")


# ====================


INPUT_PATH = "FinGPTR1_pipeline/models/NoMLPandGradUnfreeze"
OUTPUT_PATH = "models/NoMLPandGradUnfreezeLoRA"

sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")

# ====================

INPUT_PATH = "FinGPTR1_pipeline/models/NoMLPandGradUnfreeze"
OUTPUT_PATH = "models/NoMLPandGradUnfreezeLoRAWhole"

sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_', 'embeddings', 'classifier'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")


# ====================


model_name = "bert-base-uncased"
OUTPUT_PATH = "models/BertLoRA"

sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")

# ====================

model_name = "bert-base-uncased"
OUTPUT_PATH = "models/BertLoRAWhole"

sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_', 'embeddings', 'classifier'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")