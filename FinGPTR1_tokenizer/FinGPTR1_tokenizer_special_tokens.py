import torch
from torch import nn
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from tokenization.preprocess_text import preprocess_text

data =
preprocessed_data = preprocess_text(data, only_special_tokens = True)
special_tokens = ["<SON>", "<VAL>", "<EON>"]

base_model = 
base_tokenizer = 

base_tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
base_model.resize_token_embeddings(len(tokenizer))