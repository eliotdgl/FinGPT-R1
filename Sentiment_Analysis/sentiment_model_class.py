import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, AutoConfig
)
from datasets import Dataset
import numpy as np
from sklearn.metrics import accuracy_score
import os
import pandas as pd
import datetime
import glob
import shutil
import torch.nn.functional as F
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenization.preprocess_text import preprocess_text

class Sentiment_Analysis_Model:
    def __init__(self, model_name=None, label_map=None, load_model=False, num_label=3):
        self.label_map = label_map or {-1: 2, 0: 0, 1: 1}
        self.inverse_label_map = {v: k for k, v in self.label_map.items()}
        self.label_names = ["neutral", "positive", "negative"]

        self.model_name = model_name

        if not load_model and model_name is not None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name + "/tokenizer")
            self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name + "/model", num_labels=len(self.label_map)
            )
            if self.tokenizer.pad_token is None:
                self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                self.model.resize_token_embeddings(len(self.tokenizer))
                self.model.config.pad_token_id = self.tokenizer.pad_token_id

            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["query", "key", "value"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
            self.model = get_peft_model(self.model, lora_config)  # Only wrap here
        elif not load_model and model_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
            self.model = AutoModelForSequenceClassification.from_pretrained(
            "yiyanghkust/finbert-tone", num_labels=len(self.label_map)
            )
            lora_config = LoraConfig(
                r=8,
                lora_alpha=32,
                target_modules=["query", "key", "value"],
                lora_dropout=0.05,
                bias="none",
                task_type=TaskType.SEQ_CLS
            )
            self.model = get_peft_model(self.model, lora_config)
        else:
            self.tokenizer = None
            self.model = None

        
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
            new_examples = [{"text": preprocess_text(ex["text"])[0], "label": self.label_map[ex["label"]]} for ex in raw_examples]
        elif label_values.issubset({'neutral', 'positive', 'negative'}):
            mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
            new_examples = [{"text": preprocess_text(ex["text"])[0], "label": mapping[ex["label"]]} for ex in raw_examples]
        elif label_values.issubset({0, 1, 2}):
            new_examples = [{"text": preprocess_text(ex["text"])[0], "label": ex["label"]} for ex in raw_examples]
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

    def compute_metrics(self, pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    def train(self, dataset, output_dir="Sentiment_Analysis/models/sft-sentiment-model", epochs=3, batch_size=8):
        print("\nTraining started\n")
        if not self.model or not self.tokenizer:
            raise RuntimeError("\nModel is not loaded/initialized. Call load() first or initialize a new one.\n")
        dataset_splits = dataset.train_test_split(test_size=0.2)
            
            
        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            save_strategy="epoch",
            per_device_train_batch_size=batch_size,
            num_train_epochs=epochs,
            weight_decay=0.01,
            logging_dir=os.path.join(output_dir, "logs"),
            logging_steps=10,
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            greater_is_better=True
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset_splits["train"],
            eval_dataset=dataset_splits["test"],
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
        )

        trainer.train()
    def save(self, base_path="Sentiment_Analysis/models/sft-sentiment-model", timestamp_name=None, keep_last=3):
        """
        Save the current model (with LoRA adapters, if any) and tokenizer to a timestamped folder.
        Keeps only the `keep_last` most recent saved versions.
        """
    # 1. Determine timestamp
        if timestamp_name is None:
            timestamp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    # 2. Build save path and create directory
        save_path = f"{base_path}_{timestamp_name}"
        os.makedirs(save_path, exist_ok=True)

    # 3. Save model weights and adapters
        if isinstance(self.model, PeftModel):
        # Save base model
            self.model.base_model.save_pretrained(save_path)
        # Save LoRA adapters
            self.model.save_pretrained(save_path)
        else:
        # Standard model save
            self.model.save_pretrained(save_path)

    # 4. Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        print(f"\nModel and tokenizer saved to: {save_path}\n")

    # 5. Cleanup old versions
        pattern = f"{base_path}_*"
        saved_versions = sorted(glob.glob(pattern), reverse=True)
    # Keep only the most recent `keep_last` folders
        for old_path in saved_versions[keep_last:]:
            try:
                shutil.rmtree(old_path)
                print(f"Deleted old model directory: {old_path}")
            except Exception as e:
                print(f"Warning: could not delete {old_path}: {e}")


    def load(self, base_path="Sentiment_Analysis/models/sft-sentiment-model"):
    # 1. Locate the most recent version
        pattern = base_path + "_*"
        saved_versions = sorted(glob.glob(pattern), reverse=True)
        if not saved_versions:
            raise FileNotFoundError(f"\nNo saved model found at {base_path}\n")
        latest_version = saved_versions[0]
        print(f"\nLoading the latest model from: {latest_version}\n")

    # 2. Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(latest_version)

    # 3. Load config + base model
        config = AutoConfig.from_pretrained(latest_version)
        config.num_labels = len(self.label_map)
        base_model = AutoModelForSequenceClassification.from_pretrained(
        latest_version, config=config
        )

    # 4. Load PEFT adapters (restores your trained LoRA weights)
        self.model = PeftModel.from_pretrained(base_model, latest_version)

        print("\nModel and LoRA adapters successfully loaded.\n")

    def predict(self, text):
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("\nModel is not loaded. Call load() first.\n")
    
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
        with torch.no_grad():
            logits = self.model(**inputs).logits

    # Convert logits to probabilities
        probabilities = F.softmax(logits, dim=1).squeeze().tolist()

    # Get predicted class index and label
        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = self.label_names[pred_id]
        return probabilities, pred_label
    
