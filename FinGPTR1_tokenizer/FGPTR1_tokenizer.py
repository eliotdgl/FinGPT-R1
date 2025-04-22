import torch
from torch import nn
import os
import json
from transformers import LlamaTokenizer, LlamaForCausalLM

from FinGPTR1_tokenizer.training.FinGPTR1_tokenizer_training import FGPTR1_training
from FinGPTR1_tokenizer.custom_embeddings import CustomEmbeddings
from tokenization.preprocess_text import preprocess_text


class FinGPTR1_Tokenizer(nn.Module):
    def __init__(self, base_model: str = "openlm-research/open_llama_3b",
                 embeddings_path: str = "FinGPTR1_tokenizer/saved/custom_embeddings.pt",
                 train: bool = False):
        super(FinGPTR1_Tokenizer, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)

        if not base_model:
            base_model = "openlm-research/open_llama_3b"
        if not os.path.exists("FinGPTR1_tokenizer/saved/custom_embeddings.pt") or not os.path.exists("FinGPTR1_tokenizer/saved/custom_embeddings_meta.json") or train:
            print("FinGPTR1 Tokenizer not pretrained, training from default base: openlm-research/open_llama_3b")
            FGPTR1_training(base_model, embeddings_path, self.device)

        with open("FinGPTR1_tokenizer/saved/custom_embeddings_meta.json", "r") as f:
            metadata = json.load(f)

        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(base_model).to(self.device)
        
        model.resize_token_embeddings(metadata["new_vocab_len"])
        embedding_layer = model.get_input_embeddings()

        self.custom_embeddings = CustomEmbeddings(
            original_embeddings=embedding_layer,
            old_vocab_len=metadata["old_vocab_len"],
            len_vocab_added_stocks_fin=metadata["len_vocab_added_stocks_fin"],
            new_vocab_len=metadata["new_vocab_len"],
            device=self.device
        ).to(self.device)
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