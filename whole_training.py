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

train_jobs = [
    {
        "input": "FinGPTR1_pipeline/models/NoMLP",
        "output": "models/NoMLPLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": "FinGPTR1_pipeline/models/NoMLP",
        "output": "models/NoMLPLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    },
    {
        "input": "FinGPTR1_pipeline/models/NoMLPandGradUnfreeze",
        "output": "models/NoMLPandGradUnfreezeLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": "FinGPTR1_pipeline/models/NoMLPandGradUnfreeze",
        "output": "models/NoMLPandGradUnfreezeLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    },
    {
        "input": None,
        "output": "models/BertLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": None,
        "output": "models/BertLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    },
]


for job in train_jobs:
    print(f"\nTraining: {job['output']}")
    sentiment_model = Sentiment_Analysis_Model(model_name=job["input"])
    print(f"Model LOADED")
    sentiment_model.train(dataset_train, unfreeze_layers=job["unfreeze"])
    print(f"Model TRAINED")
    sentiment_model.save(base_path=job["output"], timestamp_name="1", keep_last=3)
    print(f"Model SAVED to: {job['output']}")