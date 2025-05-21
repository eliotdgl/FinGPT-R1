import torch
from torch import nn
import os
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from FinGPTR1_tokenizer.training.training_process import FGPTR1_training
from FinGPTR1_tokenizer.custom_embeddings import CustomEmbeddings
from tokenization.preprocess_text import preprocess_text


class FinGPTR1_Tokenizer(nn.Module):
    def __init__(self, PATH: str,
                base_model: str = "yiyanghkust/finbert-tone",
                 task: str = "sentiment analysis",
                 train: bool = False):
        super(FinGPTR1_Tokenizer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        if 'NoMLP' in PATH:
            self.With_MLP = False
        else:
            self.With_MLP = True

        if not base_model:
            base_model = "yiyanghkust/finbert-tone"
        if not os.path.exists(PATH + "/custom_embeddings/custom_embeddings.pt") or not os.path.exists(PATH + "/custom_embeddings/custom_embeddings_meta.json") or train:
            print("FinGPTR1 Tokenizer not pretrained, training from default base: yiyanghkust/finbert-tone")
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

        if task.lower() not in ["generation", "sentiment analysis"]:
            raise ValueError(f"Unknown task: {task}. Try one of these: generation / sentiment analysis")
        else:
            self.task = task
        
        self.model.eval()


    def forward(self, corpus):
        if isinstance(corpus, str):
            corpus = [corpus]
        preprocessed_corpus, corpus_numbers_dict = zip(*[preprocess_text(text) for text in corpus])
        
        inputs_corpus = self.tokenizer(list(preprocessed_corpus), padding=True, truncation=True, return_tensors="pt")
        input_ids_corpus = inputs_corpus["input_ids"].to(self.device)
        attention_mask = inputs_corpus["attention_mask"].to(self.device)

        with torch.no_grad():
            embeddings_customed, _ = self.custom_embeddings(input_ids_corpus, corpus_numbers_dict)

        if self.task.lower() == "generation":
            # For generation tasks, use the model's `generate` method to create text
            # Pass input_ids to the model and use the output
            outputs = self.model.generate(inputs_embeds=embeddings_customed, max_length=50, num_return_sequences=1)
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return generated_text

        elif self.task.lower() == "sentiment analysis":
            # For classification, apply the classifier head
            outputs = self.model(inputs_embeds=embeddings_customed, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predicted_class = torch.argmax(probs, dim=-1).item()
            return probs, predicted_class
    
        pass


    def tokenize(self, corpus):
        if isinstance(corpus, str):
            corpus = [corpus]
        preprocessed_corpus, corpus_numbers_dict = zip(*[preprocess_text(text) for text in corpus])
        
        return self.tokenizer(list(preprocessed_corpus), padding=True, truncation=True, return_tensors="pt")
        

    def get_embeddings(self, corpus):
        if isinstance(corpus, str):
            corpus = [corpus]
        preprocessed_corpus, corpus_numbers_dict = zip(*[preprocess_text(text) for text in corpus])
        
        inputs_corpus = self.tokenizer(list(preprocessed_corpus), padding=True, truncation=True, return_tensors="pt")
        input_ids_corpus = inputs_corpus["input_ids"].to(self.device)

        with torch.no_grad():
            embeddings_customed, _ = self.custom_embeddings(input_ids_corpus, corpus_numbers_dict)

        return embeddings_customed