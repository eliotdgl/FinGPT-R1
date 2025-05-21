import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import get_peft_model, LoraConfig, TaskType

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


raw_data = [
    "The company earned 42.15 million dollars last year.",
    "The population is approximately 7.8 billion people.",
    "Stock prices rose by +7.5% in Q1."
]
processed_data = [preprocess_text(text, only_special_tokens = True) for text in raw_data]


# === Step 4: Dataset ===
class SimpleDataset(Dataset):
    def __init__(self, texts, tokenizer):
        self.tokenizer = tokenizer
        self.data = texts

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.data[idx], return_tensors="pt", padding="max_length", truncation=True, max_length=128)
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        labels = input_ids.clone()
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

dataset = SimpleDataset(processed_data, base_tokenizer)


# LoRA config: adjust rank, alpha, dropout as needed
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["c_attn"],  # GPT2 key/query/value proj layers, adjust per model
)

# Wrap model with LoRA
model = get_peft_model(base_model, lora_config)

# Freeze all but LoRA params + embeddings + lm_head
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "lora" in name or "embed" in name or "lm_head" in name:
        param.requires_grad = True


# 6. Training arguments
training_args = TrainingArguments(
    output_dir="FinGPTR1_pipeline/models/lora_numlogic",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_strategy="epoch",
    logging_dir="FinGPTR1_pipeline/models/lora_numlogic/logs",
    logging_steps=10,
    save_total_limit=1
)


# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=base_tokenizer
)


# 8. Train
trainer.train()

# 9. Save
trainer.save_model("FinGPTR1_pipeline/models/lora_numlogic/model")
base_tokenizer.save_pretrained("FinGPTR1_pipeline/models/lora_numlogic/tokenizer")