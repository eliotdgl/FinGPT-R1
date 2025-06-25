"""
    == FGPTR1_tokenizer.py ==
    Defines the FinGPTR1_Tokenizer class for text preprocessing, tokenization, 
    embedding generation, and inference (sentiment analysis) 
    using a custom-trained transformer model with financial embeddings.
"""
import os
import json
import torch
from torch import nn
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from FinGPTR1_pipeline.training.training_process import FGPTR1_training
from FinGPTR1_pipeline.custom_embeddings import CustomEmbeddings
from tokenization.preprocess_text import preprocess_text


class FinGPTR1_Tokenizer(nn.Module):
    def __init__(self, PATH: str,
                 base_model: str = "bert-base-uncased",
                 task: str = "sentiment analysis",
                 train: bool = False):
        super(FinGPTR1_Tokenizer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Determine if MLP should be used based on PATH
        if 'NoMLP' in PATH:
            self.With_MLP = False
        else:
            self.With_MLP = True

        # Train the model and tokenizer if pretrained files do not exist or if explicitly requested
        if not base_model:
            base_model = "bert-base-uncased"
        if not os.path.exists(PATH + "/custom_embeddings/custom_embeddings.pt") or not os.path.exists(PATH + "/custom_embeddings/custom_embeddings_meta.json") or train:
            print("\nFinGPTR1 Tokenizer not pretrained, training from default base: bert-base-uncased\n")
            FGPTR1_training(PATH, base_model, self.With_MLP, self.device)

        with open(PATH + "/custom_embeddings/custom_embeddings_meta.json", "r") as f:
            metadata = json.load(f)
 
        self.tokenizer = AutoTokenizer.from_pretrained(PATH + '/tokenizer')
        self.model = AutoModelForSequenceClassification.from_pretrained(PATH + '/model').to(self.device)
        
        embedding_layer = self.model.get_input_embeddings()

        self.custom_embeddings = CustomEmbeddings(
            original_embeddings=embedding_layer,
            old_vocab_len=metadata["old_vocab_len"],
            len_vocab_added_stocks_fin=metadata["len_vocab_added_stocks_fin"],
            new_vocab_len=metadata["new_vocab_len"],
            WithMLP = self.With_MLP,
            device=self.device
        ).to(self.device)
        self.custom_embeddings.load_state_dict(torch.load(PATH + "/custom_embeddings/custom_embeddings.pt", map_location=self.device))
        self.custom_embeddings.eval()

        if task.lower() not in ["sentiment analysis"]:
            raise ValueError(f"Unknown task: {task}. Try one of these: sentiment analysis")
        else:
            self.task = task
        
        self.model.eval()


    def forward(self, corpus):
        """
            Forward method to perform sentiment analysis.
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        preprocessed_corpus, corpus_numbers_dict = zip(*[preprocess_text(text) for text in corpus])
        
        inputs_corpus = self.tokenizer(list(preprocessed_corpus), padding=True, truncation=True, return_tensors="pt")
        input_ids_corpus = inputs_corpus["input_ids"].to(self.device)
        attention_mask = inputs_corpus["attention_mask"].to(self.device)

        with torch.no_grad():
            embeddings_customed, _ = self.custom_embeddings(input_ids_corpus, corpus_numbers_dict)
        
        if self.task.lower() == "sentiment analysis":
            # For classification, apply the classifier head
            outputs = self.model(inputs_embeds=embeddings_customed, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            return probs, predicted_class
    

    def tokenize(self, corpus) -> dict:
        """
            Tokenizes and preprocesses the input corpus.
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        preprocessed_corpus, corpus_numbers_dict = zip(*[preprocess_text(text) for text in corpus])
        
        return self.tokenizer(list(preprocessed_corpus), padding=True, truncation=True, return_tensors="pt")
        

    def get_embeddings(self, corpus) -> torch.Tensor:
        """
            Generate custom embeddings for the given corpus.
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        preprocessed_corpus, corpus_numbers_dict = zip(*[preprocess_text(text) for text in corpus])
        
        inputs_corpus = self.tokenizer(list(preprocessed_corpus), padding=True, truncation=True, return_tensors="pt")
        input_ids_corpus = inputs_corpus["input_ids"].to(self.device)

        with torch.no_grad():
            embeddings_customed, _ = self.custom_embeddings(input_ids_corpus, corpus_numbers_dict)

        return embeddings_customed
