from huggingface_hub import login
login("")

import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM

from tokenization.preprocess_text import preprocess_text

# Load Llama tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b")

old_vocab_len = len(tokenizer)


# Import new tokens
import json 
    # Load stock indices vocabulary
with open("tokenization/vocabulary/stock_indices_vocab.json", "r") as f:
    stock_indices = list(json.load(f).values())
    # Load stock tickers vocabulary
with open("tokenization/vocabulary/stock_tickers_vocab.json", "r") as f:
    stock_tickers = json.load(f)
    # Load numerical vocabulary
with open("tokenization/vocabulary/numericals_vocab.json", "r") as f:
    num_tokens = json.load(f)
    # Load financial vocabulary
with open("data/vocabulary/financial_vocab.json", "r") as f:
    financial_tokens = json.load(f)

# Add new tokens to the tokenizer
tokenizer.add_tokens(stock_indices)
tokenizer.add_tokens(stock_tickers)
tokenizer.add_tokens(num_tokens)
tokenizer.add_tokens(financial_tokens)

new_vocab_len = len(tokenizer)


# List of indices of the new added tokens
new_token_indices = list(range(old_vocab_len, new_vocab_len))
new_token_indices_torch = torch.tensor(new_token_indices)


# Resize the model's embedding layer to accommodate the new added tokens
model.resize_token_embeddings(new_vocab_len)

# Freeze all original embeddings from Llama tokenizer to prevent them from being updated during training
for param in model.parameters():
    param.requires_grad = False


# Get embedding layer of Llama model
embedding_layer = model.get_input_embeddings()


# Define an MLP for personalized embeddings of the new added tokens
class EmbeddingMLP(nn.Module):
    def __init__(self, embedding_dim: int):
        super(EmbeddingMLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim * 2),
            nn.ReLU(),
            nn.Linear(embedding_dim * 2, embedding_dim)
        )

    def forward(self, x):
        return self.mlp(x)


# Define a custom embedding module combining original model's embeddings and personalized embeddings from EmbeddingMLP()
class CustomEmbeddings(nn.Module):
    def __init__(self, embedding_dim: int, original_embeddings: torch, new_token_indices: torch):
        super(CustomEmbeddings, self).__init__()
        self.new_token_indices = new_token_indices
        self.original_embeddings = original_embeddings
        self.mlp = EmbeddingMLP(embedding_dim)

    def forward(self, input_ids: torch):
        with torch.no_grad():
            embeddings = self.original_embeddings(input_ids)
            # Create a mask for the new added tokens
            masked_embeddings = torch.isin(input_ids, self.new_token_indices)
        # Modify the embeddings for the new added tokens only using the MLP
        embeddings[masked_embeddings] = self.mlp(embeddings[masked_embeddings])

        return embeddings


embedding_dim = model.get_input_embeddings().embedding_dim

Custom_Embeddings = CustomEmbeddings(embedding_dim, model.get_input_embeddings(), new_token_indices_torch)

optimizer = torch.optim.AdamW(Custom_Embeddings.parameters(), lr=5e-5)

from data.financial_news.NIFTY import get_data
data = get_data()
train_data = data["train"].to_pandas()
news, labels = train_data["news"], train_data["label"]

num_epochs = 3
epoch_bar = tqdm(range(num_epochs))

# Training loop to train EmbeddingMLP() 
Custom_Embeddings.train()
for epoch in epoch_bar:

    epoch_loss = 0
    for batch in dataloader:
        inputs = tokenizer(batch, return_tensors="pt")
        input_ids = inputs["input_ids"]

        optimizer.zero_grad()
        # Get the combined embeddings (original|new_added_tokens) for the tokens in the batch
        embeddings = Custom_Embeddings(input_ids)
        output = model(inputs_embeds=embeddings, labels=input_ids)
        loss = output.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch} loss: {epoch_loss}")


Custom_Embeddings.eval()
with torch.no_grad():
    # Generate new embeddings including trained embeddingsfor the new added tokens
    new_embeddings = Custom_Embeddings(new_token_indices_torch)

# Update the model's embedding layer
embedding_layer = model.get_input_embeddings()
embedding_layer.weight[old_vocab_len:] = new_embeddings.detach()

# Save the personalized tokenizer
Path = ""
tokenizer.save_pretrained(Path)

print("Tokenizer saved successfully.")