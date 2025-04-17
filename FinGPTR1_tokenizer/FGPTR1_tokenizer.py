import torch
from torch import nn
import os
from transformers import LlamaTokenizer

from FinGPTR1_tokenizer.training.FinGPTR1_tokenizer_training import FGPTR1_training
from FinGPTR1_tokenizer.custom_embeddings import CustomEmbeddings
from tokenization.preprocess_text import preprocess_text


class FinGPTR1_Tokenizer(nn.Module):
    def __init__(self, base_model: str = None, base_tokenizer: str = None,
                 data = None, 
                 tokenizer_path: str = "FinGPTR1_tokenizer_training/saved/saved_tokenizer",
                 embeddings_path: str = "FinGPTR1_tokenizer_training/saved/custom_embeddings.pt"):
        super(FinGPTR1_Tokenizer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        if base_model and base_tokenizer:
            print(f"Training FinGPTR1 Tokenizer from given base: {base_model} | {base_tokenizer}")
            FGPTR1_training(base_model, base_tokenizer, data, tokenizer_path, embeddings_path, self.device)
        elif not os.path.exists(tokenizer_path) or not os.path.exists(embeddings_path):
            print("FinGPTR1 Tokenizer not pretrained, training from default base: openlm-research/open_llama_3b")
            FGPTR1_training(None, None, data, tokenizer_path, embeddings_path, self.device)


        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        self.custom_embeddings = CustomEmbeddings().to(self.device)
        self.custom_embeddings.load_state_dict(torch.load(embeddings_path, map_location=self.device))
        self.custom_embeddings.eval()


    def forward(self, corpus):
        if isinstance(corpus, str):
            corpus = [corpus]
        preprocessed_corpus, corpus_numbers_dict = zip(*[preprocess_text(text) for text in corpus])
        
        inputs_corpus = self.tokenizer(list(preprocessed_corpus), return_tensors="pt")
        input_ids_corpus = inputs_corpus["input_ids"].to(self.device)
        attention_mask = inputs_corpus["attention_mask"].to(self.device)

        with torch.no_grad():
            embeddings_list = []
            for i in range(input_ids_corpus.size(0)):
                input_ids = input_ids_corpus[i].unsqueeze(0)
                text_dict = corpus_numbers_dict[i]
                
                embeddings = self.custom_embeddings(input_ids, text_dict)
                embeddings_list.append(embeddings)
        
        embeddings_batch = torch.cat(embeddings_list, dim=0)
        
        return embeddings_batch, attention_mask