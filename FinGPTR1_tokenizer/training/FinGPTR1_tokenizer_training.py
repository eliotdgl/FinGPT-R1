import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM

from tokenization.preprocess_text import preprocess_text

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Load Llama tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b").to(device)


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
tokenizer.add_tokens(financial_tokens)

len_vocab_added_stocks_fin = len(tokenizer)

tokenizer.add_tokens(num_tokens)

new_vocab_len = len(tokenizer)


# Resize the model's embedding layer to accommodate the new added tokens
model.resize_token_embeddings(new_vocab_len)

# Freeze all original embeddings from Llama tokenizer to prevent them from being updated during training
for param in model.parameters():
    param.requires_grad = False


# Get embedding layer of Llama model
embedding_layer = model.get_input_embeddings()


# Define a custom embedding module combining original model's embeddings and personalized embeddings
class CustomEmbeddings(nn.Module):
    def __init__(self, original_embeddings: torch, old_vocab_len: int, len_vocab_added_stocks_fin: int, new_vocab_len: int):
        super(CustomEmbeddings, self).__init__()
        self.original_embeddings = original_embeddings
        
        new_tokens_indices = list(range(old_vocab_len, new_vocab_len))

        stocks_fin_indices = list(range(old_vocab_len, len_vocab_added_stocks_fin))
        self.stocks_fin_indices_torch = torch.tensor(stocks_fin_indices).to(device)

        num_indices = list(range(len_vocab_added_stocks_fin, new_vocab_len))
        self.num_indices_torch = torch.tensor(num_indices).to(device)
        
        self.embedding_dim=original_embeddings.embedding_dim
        
        self.new_embeddings_layer = nn.Embedding(
            num_embeddings=len(new_tokens_indices),
            embedding_dim=self.embedding_dim
        )

        input_dim = 4
        self.num_mlp = nn.Sequential(
            nn.Linear(input_dim, 2*self.embedding_dim),
            nn.ReLU(),
            nn.Linear(2*self.embedding_dim, self.embedding_dim)
        )

        # Randomly initialize new embeddings according to original embeddings
        with torch.no_grad():
            mean = self.original_embeddings.weight[:old_vocab_len].mean(dim=0)
            std = self.original_embeddings.weight[:old_vocab_len].std(dim=0)

            mean_matrix = mean.unsqueeze(0).expand(len(new_tokens_indices), -1).clone()
            std_matrix = std.unsqueeze(0).expand(len(new_tokens_indices), -1).clone()
            self.new_embeddings_layer.weight.data = torch.normal(mean=mean_matrix, std=std_matrix)

    def forward(self, input_ids: torch, num_dict: dict):
        embeddings = self.original_embeddings(input_ids).clone()
        # Create a mask for the new added tokens
        masked_stocks_fin_embeddings = torch.isin(input_ids, self.stocks_fin_indices_torch)
        masked_num_embeddings = torch.isin(input_ids, self.num_indices_torch)
        # Modify the embeddings for the new added tokens only
        if masked_stocks_fin_embeddings.any():
            new_stocks_fin_input_ids = input_ids[masked_stocks_fin_embeddings] - self.stocks_fin_indices_torch[0]
            embeddings[masked_stocks_fin_embeddings] = self.new_embeddings_layer(new_stocks_fin_input_ids)
        if masked_num_embeddings.any():
            new_num_input_ids = input_ids[masked_num_embeddings] - self.num_indices_torch[0]
            embeddings_from_emblayer = self.new_embeddings_layer(new_num_input_ids)

            num_features = torch.stack([torch.tensor(list(num_dict[i].values()), dtype=torch.float32) for i in range(masked_num_embeddings.sum())], dim=0).to(input_ids.device)
            embeddings_from_mlp = self.num_mlp(num_features)

            embeddings[masked_num_embeddings] = embeddings_from_emblayer + embeddings_from_mlp

        return embeddings


Custom_Embeddings = CustomEmbeddings(embedding_layer, old_vocab_len, len_vocab_added_stocks_fin, new_vocab_len).to(device)
optimizer = torch.optim.AdamW(
    list(Custom_Embeddings.new_embeddings_layer.parameters()) +
    list(Custom_Embeddings.num_mlp.parameters()),
    lr=5e-5
)


"""
from data.financial_news.NIFTY import get_data
data = get_data()
train_data = data["train"].to_pandas()
news, labels = train_data["news"], train_data["label"]
"""

dataloader = {}

num_epochs = 20
epoch_bar = tqdm(range(num_epochs))

# Training loop to train EmbeddingMLP() 
Custom_Embeddings.train()
for epoch in epoch_bar:

    epoch_loss = 0
    for batch in dataloader:
        preprocessed_batch, batch_numbers_dict = preprocess_text(batch)
        inputs = tokenizer(preprocessed_batch, return_tensors="pt")
        input_ids = inputs["input_ids"].to(device)

        optimizer.zero_grad()
        # Get the personalized embeddings (original|new_added_tokens) for the tokens in the batch
        embeddings = Custom_Embeddings(input_ids, batch_numbers_dict)
        output = model(inputs_embeds=embeddings, labels=input_ids)
        loss = output.loss
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
    print(f"Epoch {epoch} loss: {epoch_loss}")


with torch.no_grad():
    # Update the model's embedding layer
    embedding_layer.weight[old_vocab_len:] = Custom_Embeddings.new_embeddings_layer.weight.clone()

# Save the personalized tokenizer
Path = "./saved_tokenizer"
tokenizer.save_pretrained(Path)
torch.save(Custom_Embeddings.state_dict(), "custom_embeddings.pt")

print("Tokenizer saved successfully")