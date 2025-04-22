import torch
from torch import nn

# Define a custom embedding module combining original model's embeddings and personalized embeddings
class CustomEmbeddings(nn.Module):
    def __init__(self, original_embeddings: torch = None, old_vocab_len: int = None, len_vocab_added_stocks_fin: int = None, new_vocab_len: int = None, device = None):
        super(CustomEmbeddings, self).__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if original_embeddings is not None and old_vocab_len is not None and len_vocab_added_stocks_fin is not None and new_vocab_len is not None:
            self.original_embeddings = original_embeddings
            
            new_tokens_indices = list(range(old_vocab_len, new_vocab_len))

            stocks_fin_indices = list(range(old_vocab_len, len_vocab_added_stocks_fin))
            self.stocks_fin_indices_torch = torch.tensor(stocks_fin_indices).to(device)

            num_indices = list(range(len_vocab_added_stocks_fin, new_vocab_len))
            self.num_indices_torch = torch.tensor(num_indices).to(device)
            
            self.embedding_dim = original_embeddings.embedding_dim
            
            self.new_embeddings_layer = nn.Embedding(
                num_embeddings=len(new_tokens_indices),
                embedding_dim=self.embedding_dim
            )

            input_dim = 4
            self.num_mlp = nn.Sequential(
                nn.Linear(input_dim, 2*self.embedding_dim),
                nn.Tanh(),
                nn.Linear(2*self.embedding_dim, self.embedding_dim)
            )

            # Randomly initialize new embeddings according to original embeddings
            with torch.no_grad():
                mean = self.original_embeddings.weight[:old_vocab_len].mean(dim=0)
                std = self.original_embeddings.weight[:old_vocab_len].std(dim=0)

                mean_matrix = mean.unsqueeze(0).expand(len(new_tokens_indices), -1).clone()
                std_matrix = std.unsqueeze(0).expand(len(new_tokens_indices), -1).clone()
                self.new_embeddings_layer.weight.data = torch.normal(mean=mean_matrix, std=std_matrix)
        
        else:
            # If no embeddings are passed, it is assumed we are loading a pretrained model
            self.original_embeddings = None
            self.new_embeddings_layer = None
            self.num_mlp = None
            self.stocks_fin_indices_torch = None
            self.num_indices_torch = None


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