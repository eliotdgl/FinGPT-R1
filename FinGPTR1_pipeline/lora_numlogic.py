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

# Add the project root folder to sys.path so 'tokenization' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenization.preprocess_text import preprocess_text


special_tokens = ["<SON>", "<VAL>", "<EON>"]

model_name = "gpt2"
base_tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

base_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
base_tokenizer.pad_token = base_tokenizer.eos_token
base_model.config.pad_token_id = base_tokenizer.pad_token_id
base_model.resize_token_embeddings(len(base_tokenizer))

raw_data = load_dataset(
    'csv',
    data_files='data/sentiment_analysis_train/FinancialPhraseBank-v1.0/Sentences_AllAgree_processed.csv'
)
dataset = raw_data['train']

def contains_numbers(example):
    return bool(re.search(r'\d', example['Sentence']))

filtered_dataset = dataset.filter(contains_numbers)
print(f"Filtered dataset size: {len(filtered_dataset)}")

def add_special_tokens(example):
    return {"text": preprocess_text(example["Sentence"], only_special_tokens=True)}

processed_data = filtered_dataset.map(add_special_tokens)


id2label = {0: "neutral", 1: "positive", 2: "negative"}
label2id = {"neutral": 0, "positive": 1, "negative": 2}

dataset = pd.read_csv('data/local_data/test_all_agree.csv', header=None, names=["Sentence", "Label"], skiprows=1)

def tokenize(example):
    enc = base_tokenizer(example["Sentence"], truncation=True, padding="max_length", max_length=128)
    enc["labels"] = label2id[example["Label"]]
    return enc

tokenized_dataset = processed_data.map(tokenize,)
tokenized_dataset.set_format("torch")

# LoRA config: adjust rank, alpha, dropout as needed
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["query", "key", "value"]  # GPT2 key/query/value proj layers, adjust per model
)

model = get_peft_model(base_model, lora_config)

training_args = TrainingArguments(
    output_dir="FinGPTR1_pipeline/models/LoRA_numlogic",
    per_device_train_batch_size=16,
    num_train_epochs=3,
    logging_dir=os.path.join("FinGPTR1_pipeline/models/LoRA_numlogic", "logs"),
    logging_steps=10,
    save_total_limit=2
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=base_tokenizer
)

trainer.train()

full_model = model.merge_and_unload()
full_model.save_pretrained("FinGPTR1_pipeline/models/LoRA_numlogic/model")
base_tokenizer.save_pretrained("FinGPTR1_pipeline/models/LoRA_numlogic/tokenizer")