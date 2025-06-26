"""
  == cache_models.py ==
  Loads and caches pretrained tokenizers and models from Hugging Face for faster subsequent use.

  Models included:
  - BERT: bert-base-uncased (base model)
  - FinBERT: yiyanghkust/finbert-tone (baseline)
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "bert-base-uncased"
base_tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name)

model_name = "yiyanghkust/finbert-tone"
base_tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSequenceClassification.from_pretrained(model_name)
