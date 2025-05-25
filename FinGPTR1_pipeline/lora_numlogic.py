import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    AutoModelForSequenceClassification,
    pipeline
)
from peft import get_peft_model, LoraConfig, TaskType
import re

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenization.preprocess_text import preprocess_text
from Sentiment_Analysis.sentiment_model_class import Sentiment_Analysis_Model
from Sentiment_Analysis.controlled_environment import TextDataGenerator
from Sentiment_Analysis.data_loader import get_train_test_split


special_tokens = ["<SON>", "<VAL>", "<EON>"]


# Load the CSV file
print("\nLoading dataset...\n")

generator=TextDataGenerator(number_of_episode=5)
generated_data=generator.generate_batch()

sentiment_data = Sentiment_Analysis_Model()
dataset_train = sentiment_data.prepare_dataset(generated_data) 

print("\nDataset LOADED\n")


def prepare_dataset(self, raw_input):
    # Load raw data
    if isinstance(raw_input, list) and len(raw_input) == 2:
        texts, labels = raw_input
        labels = [int(label) for label in labels]
        raw_examples = [{"text": text, "label": label} for text, label in zip(texts, labels)]
    elif isinstance(raw_input, str) and raw_input.endswith(".csv"):
        df = pd.read_csv(raw_input, header=None, names=["text", "label"], skiprows=1)
        raw_examples = df.to_dict(orient="records")
    elif isinstance(raw_input, list):
        raw_examples = raw_input
    else:
        raise ValueError("Input must be a CSV path or list of dicts/lists.")

    # Remap labels
    label_values = set(ex["label"] for ex in raw_examples)
    if label_values.issubset(set(self.label_map.keys())):
        new_examples = [{"text": preprocess_text(ex["text"], only_special_tokens=True), "label": self.label_map[ex["label"]]} for ex in raw_examples]
    elif label_values.issubset({'neutral', 'positive', 'negative'}):
        mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
        new_examples = [{"text": preprocess_text(ex["text"], only_special_tokens=True), "label": mapping[ex["label"]]} for ex in raw_examples]
    elif label_values.issubset({0, 1, 2}):
        new_examples = [{"text": preprocess_text(ex["text"], only_special_tokens=True), "label": ex["label"]} for ex in raw_examples]
    else:
        raise ValueError(f"Unexpected label values: {label_values}")

    dataset = Dataset.from_list(new_examples)
    return dataset.map(self._tokenize)


def _tokenize(self, example):
    tok = self.tokenizer(
    example["text"],
    truncation=True,
    padding="max_length",
    max_length=128,
    )
    tok["labels"] = example["label"]
    return tok


# ===== Training =====

model_name = "yiyanghkust/finbert-tone"
base_tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

base_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model.config.pad_token_id = base_tokenizer.pad_token_id
base_model.resize_token_embeddings(len(base_tokenizer))

base_model.save_pretrained("FinGPTR1_pipeline/models/NumLogic/model")
base_tokenizer.save_pretrained("FinGPTR1_pipeline/models/NumLogic/tokenizer")


# ====================

print(f"\nTraining: NumLogicLoRA")

INPUT_PATH = "FinGPTR1_pipeline/models/NumLogic"
OUTPUT_PATH = "models/NumLogicLoRA"


sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")

# ====================

print(f"\nTraining: NumLogicLoRAWhole")

OUTPUT_PATH = "models/NumLogicLoRAWhole"

sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_', 'embeddings', 'classifier'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")