"""
    == sentiment_model_class.py ==
    Sentiment Analysis Model with LoRA fine-tuning
    
    Defines a class `Sentiment_Analysis_Model` for training,
    saving, loading, and inference of a sentiment analysis model
    based on Hugging Face Transformers and PEFT (LoRA) adapters.

    Usage:
        Instantiate the `Sentiment_Analysis_Model` class,
        call `train()` with your dataset,
        save the model with `save()`,
        load it later with `load()`,
        and predict using `predict()`.
    
        Example:
            model = Sentiment_Analysis_Model(model_name="bert-base-uncased")
            dataset = model.prepare_dataset("path/to/data.csv")
            model.train(dataset)
            model.save()
            probs, label = model.predict(`sentence`)
"""
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset
from sklearn.metrics import accuracy_score
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, AutoConfig
)
from peft import get_peft_model, LoraConfig, TaskType, PeftModel
import glob
import shutil
import datetime

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tokenization.preprocess_text import preprocess_text


class Sentiment_Analysis_Model:
    def __init__(self, model_name = None, label_map = None, load_model: bool = False, num_label: int = 3):
        """
            Initalize an instance of the Sentiment_Analysis_Model class, either creates a new model 
            or loads an existing one.
        """
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
            self.model = get_peft_model(self.model, lora_config)

        elif not load_model and model_name is None:
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=len(self.label_map)
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

        
    def prepare_dataset(self, raw_input, preprocess: bool = True, numlogic_model: bool = False):
        """
            Prepares a HuggingFace dataset from raw input.
        """
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

        if preprocess:
            if numlogic_model:
                label_values = set(ex["label"] for ex in raw_examples)
                if label_values.issubset(set(self.label_map.keys())):
                    new_examples = [{"text": preprocess_text(ex["text"], only_special_tokens=True)[0], "label": self.label_map[ex["label"]]} for ex in raw_examples]
                elif label_values.issubset({'neutral', 'positive', 'negative'}):
                    mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
                    new_examples = [{"text": preprocess_text(ex["text"], only_special_tokens=True)[0], "label": mapping[ex["label"]]} for ex in raw_examples]
                elif label_values.issubset({0, 1, 2}):
                    new_examples = [{"text": preprocess_text(ex["text"], only_special_tokens=True)[0], "label": ex["label"]} for ex in raw_examples]
                else:
                    raise ValueError(f"Unexpected label values: {label_values}")
            else:
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
        else:
            label_values = set(ex["label"] for ex in raw_examples)
            if label_values.issubset(set(self.label_map.keys())):
                new_examples = [{"text": ex["text"], "label": self.label_map[ex["label"]]} for ex in raw_examples]
            elif label_values.issubset({'neutral', 'positive', 'negative'}):
                mapping = {'neutral': 0, 'positive': 1, 'negative': 2}
                new_examples = [{"text": ex["text"], "label": mapping[ex["label"]]} for ex in raw_examples]
            elif label_values.issubset({0, 1, 2}):
                new_examples = [{"text": ex["text"], "label": ex["label"]} for ex in raw_examples]
            else:
                raise ValueError(f"Unexpected label values: {label_values}")

        dataset = Dataset.from_list(new_examples)
        return dataset


    def _tokenize(self, example):
        """
            Tokenize a single example for model input.
        """
        tok = self.tokenizer(
            example["text"],
            truncation=True,
            padding="max_length",
            max_length=128,
        )
        tok["labels"] = example["label"]
        return tok

    
    def compute_metrics(self, pred) -> dict:
        """
            Computes accuracy from predictions and true labels.
        """
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

    
    def train(self, input_dataset, output_dir: str = "sentiment_analysis/logs/sft-sentiment-model", unfreeze_layers: list = ['lora_'],
              epochs: int = 3, batch_size: int = 32) -> None:
        """
            Trains the sentiment analysis model.
        """
        dataset = input_dataset.map(self._tokenize)
        if not self.model or not self.tokenizer:
            raise RuntimeError("\nModel is not loaded/initialized. Call load() first or initialize a new one.\n")
        dataset_splits = dataset.train_test_split(test_size=0.2)

        # Freeze all parameters
        for name, param in self.model.named_parameters():
            param.requires_grad = False
        # Unfreeze specified layers
        for name, param in self.model.named_parameters():
            if any(key in name for key in unfreeze_layers):
                param.requires_grad = True
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params

        print(f"\nProportion trainable parameters: {trainable_ratio * 100:.2f}%\n")
            
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


    def save(self, base_path: str = "sentiment_analysis/models/sft-sentiment-model", timestamp_name = None, keep_last: int = 3) -> None:
        """
            Saves the current model (with LoRA adapters, if any) and tokenizer to a timestamped folder.
            Keeps only the `keep_last` most recent saved versions.
        """
        if timestamp_name is None:
            timestamp_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        save_path = f"{base_path}_{timestamp_name}"
        os.makedirs(save_path, exist_ok=True)

        if isinstance(self.model, PeftModel):
        # Save base model
            self.model.base_model.save_pretrained(save_path)
        # Save config
            self.model.base_model.config.save_pretrained(save_path) 
        # Save LoRA adapters
            self.model.save_pretrained(save_path)
        else:
        # Standard model save
            self.model.save_pretrained(save_path)

        # Save tokenizer
        self.tokenizer.save_pretrained(save_path)

        print(f"\nModel and tokenizer saved to: {save_path}\n")

        pattern = f"{base_path}_*"
        saved_versions = sorted(glob.glob(pattern), reverse=True)
        for old_path in saved_versions[keep_last:]:
            try:
                shutil.rmtree(old_path)
                print(f"Deleted old model directory: {old_path}")
            except Exception as e:
                print(f"Warning: could not delete {old_path}: {e}")


    def load(self, base_path: str = "sentiment_analysis/models/sft-sentiment-model") -> None:
        """
            Loads the most recent saved model and tokenizer including LoRA adapters.
        """
        pattern = base_path + "_*"
        saved_versions = sorted(glob.glob(pattern), reverse=True)
        if not saved_versions:
            raise FileNotFoundError(f"\nNo saved model found at {base_path}\n")
        latest_version = saved_versions[0]
        print(f"\nLoading the latest model from: {latest_version}\n")

        self.tokenizer = AutoTokenizer.from_pretrained(latest_version)

        config = AutoConfig.from_pretrained(latest_version, local_files_only=True)
        config.num_labels = len(self.label_map)
        base_model = AutoModelForSequenceClassification.from_pretrained(
        latest_version, config=config, local_files_only=True
        )

        self.model = PeftModel.from_pretrained(base_model, latest_version, local_files_only=True)

        print("\nModel and LoRA adapters successfully loaded.\n")


    def predict(self, text: str) -> tuple:
        """"
            Predicts sentiment label and probabilities for a given text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("\nModel is not loaded. Call load() first.\n")
    
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    
        with torch.no_grad():
            logits = self.model(**inputs).logits

        probabilities = F.softmax(logits, dim=1).squeeze().tolist()

        pred_id = torch.argmax(logits, dim=1).item()
        pred_label = self.label_names[pred_id]
        return probabilities, pred_label
