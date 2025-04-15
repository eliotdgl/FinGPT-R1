import torch
from torch import nn
from transformers import LlamaTokenizer
from FinGPTR1_tokenizer.training.FinGPTR1_tokenizer_training import CustomEmbeddings
from tokenization.preprocess_text import preprocess_text

class FinGPTR1_Tokenizer(nn.Module):
    def __init__(self):
        super(FinGPTR1_Tokenizer, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = LlamaTokenizer.from_pretrained("FinGPTR1_tokenizer_training/saved/saved_tokenizer")
        self.custom_embeddings = CustomEmbeddings().to(self.device)
        self.custom_embeddings.load_state_dict(torch.load("custom_embeddings.pt", map_location=self.device))
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