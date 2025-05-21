import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import get_peft_model, LoraConfig, TaskType

import sys
import os

# Add the project root folder to sys.path so 'tokenization' can be found
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from tokenization.preprocess_text import preprocess_text


data = [
    "The company earned 42.15 million dollars last year.",
    "The population is approximately 7.8 billion people.",
    "Stock prices rose by +7.5% in Q1."
]

preprocessed_data = [preprocess_text(text, only_special_tokens = True) for text in data]
special_tokens = ["<SON>", "<VAL>", "<EON>"]

model_name = "yiyanghkust/finbert-tone"
base_tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForCausalLM.from_pretrained(model_name)

base_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
base_model.resize_token_embeddings(len(base_tokenizer))

# LoRA config: adjust rank, alpha, dropout as needed
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["query", "key", "value"],  # GPT2 key/query/value proj layers, adjust per model
)

# Wrap model with LoRA
model = get_peft_model(base_model, lora_config)

# Freeze all but LoRA params + embeddings + lm_head
for param in model.parameters():
    param.requires_grad = False
for name, param in model.named_parameters():
    if "lora" in name or "embed" in name or "lm_head" in name:
        param.requires_grad = True

# 4. Dataset class
class NumberDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=128):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encodings = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].squeeze()
        attention_mask = encodings["attention_mask"].squeeze()
        labels = input_ids.clone()
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

dataset = NumberDataset(preprocessed_data, base_tokenizer)

# 5. Data collator to handle padding (optional here since we pad to max_length)
data_collator = DataCollatorForLanguageModeling(tokenizer=base_tokenizer, mlm=False)

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=2,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=50,
    learning_rate=5e-5,
    weight_decay=0.01,
    save_total_limit=1,
    remove_unused_columns=False,
    fp16=torch.cuda.is_available(),
)

# 7. Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    tokenizer=base_tokenizer,
    data_collator=data_collator,
)

# 8. Train
trainer.train()

# 9. Save
trainer.save_model("FinGPTR1_pipeline/models/lora_numlogic_model/model")
base_tokenizer.save_pretrained("FinGPTR1_pipeline/models/lora_numlogic_model/tokenizer")