from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, AutoConfig
)
import pandas as pd
import os
from peft import get_peft_model, LoraConfig, TaskType, PeftModel, PeftConfig
from datasets import load_dataset, Dataset
import numpy as np
from sklearn.metrics import accuracy_score


def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {"accuracy": accuracy_score(labels, preds)}

def tokenize(example):
    tok = base_tokenizer(
    example["Sentence"],
    truncation=True,
    padding="longest",
    max_length=128,
    )
    tok["labels"] = mapping[example["Label"]]
    return tok


def bert_train():
    base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    mapping = {'neutral': 0, 'positive': 1, 'negative': 2}

    train_data = load_dataset('csv', data_files='data/local_data/train_all_agree.csv')

    dataset = train_data.map(tokenize)['train']
    dataset_splits = dataset.train_test_split(test_size=0.2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    epochs = 3
    batch_size = 16

    unfreeze_layers = ['lora_']
    lora_model = get_peft_model(base_model, lora_config)
    output_dir = 'BERT/models/BertLoRA_'

    for name, param in lora_model.named_parameters():
        param.requires_grad = False
            
    for name, param in lora_model.named_parameters():
        if any(key in name for key in unfreeze_layers):
            param.requires_grad = True

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
        model=lora_model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        tokenizer=base_tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    lora_model.base_model.save_pretrained(output_dir)
    lora_model.save_pretrained(output_dir)
    base_tokenizer.save_pretrained(output_dir)

    pass


def bertec_train():
    base_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
    base_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    mapping = {'neutral': 0, 'positive': 1, 'negative': 2}

    train_data = load_dataset('csv', data_files='data/local_data/train_all_agree.csv')

    dataset = train_data.map(tokenize)['train']
    dataset_splits = dataset.train_test_split(test_size=0.2)

    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["query", "key", "value"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_CLS
    )

    epochs = 3
    batch_size = 16
    
    unfreeze_layers = ["lora_", "embeddings", "classifier"]
    lora_model = get_peft_model(base_model, lora_config)
    output_dir = 'BERT/models/BertLoRAWhole_'

    for name, param in lora_model.named_parameters():
        param.requires_grad = False
            
    for name, param in lora_model.named_parameters():
        if any(key in name for key in unfreeze_layers):
            param.requires_grad = True

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
        model=lora_model,
        args=training_args,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["test"],
        tokenizer=base_tokenizer,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    lora_model.base_model.save_pretrained(output_dir)
    lora_model.save_pretrained(output_dir)
    base_tokenizer.save_pretrained(output_dir)

    pass