import torch
from torch import nn


# Define a custom embedding module combining original model's embeddings and personalized embeddings
class CustomEmbeddings(nn.Module):
    def __init__(self, original_embeddings: torch, old_vocab_len: int, len_vocab_added_stocks_fin: int, new_vocab_len: int = None, device = None):
        super(CustomEmbeddings, self).__init__()

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
    

        self.original_embeddings = original_embeddings
        
        new_tokens_indices = list(range(old_vocab_len, new_vocab_len))

        stocks_fin_indices = list(range(old_vocab_len, len_vocab_added_stocks_fin))
        self.stocks_fin_indices_torch = torch.tensor(stocks_fin_indices).to(self.device)

        num_indices = list(range(len_vocab_added_stocks_fin, new_vocab_len))
        self.num_indices_torch = torch.tensor(num_indices).to(self.device)
        
        self.embedding_dim = original_embeddings.embedding_dim

        self.new_embeddings_layer = nn.Embedding(
            num_embeddings=len(new_tokens_indices),
            embedding_dim=self.embedding_dim
        )

        self.unit_embed = nn.Embedding(
            num_embeddings=6, 
            embedding_dim=2
        )

        self.num_mlp = nn.Sequential(
            nn.Linear(3, 4*self.embedding_dim),
            nn.GELU(),
            nn.Linear(4*self.embedding_dim, self.embedding_dim)
        )

        # Randomly initialize new embeddings according to original embeddings
        mean = self.original_embeddings.weight[:old_vocab_len].mean(dim=0)
        std = self.original_embeddings.weight[:old_vocab_len].std(dim=0)

        mean_matrix = mean.unsqueeze(0).expand(len(new_tokens_indices), -1).clone()
        std_matrix = std.unsqueeze(0).expand(len(new_tokens_indices), -1).clone()
        self.new_embeddings_layer.weight.data = torch.normal(mean=mean_matrix, std=std_matrix)


        self.new_token_ids = list(range(old_vocab_len, new_vocab_len))

        # Build mapping: token_id → index in new_embeddings_layer
        self.token_id_to_new_embedding_idx_tensor = torch.full((new_vocab_len,), -1, dtype=torch.long).to(self.device)
        for i, token_id in enumerate(self.new_token_ids):
            self.token_id_to_new_embedding_idx_tensor[token_id] = i
        
        self.to(self.device)
        
    def enhance_num_dict(self, numericals_dicts):
        batch_values = []
        batch_units = []

        for num_dict in numericals_dicts:
            for v in num_dict.values():
                batch_values.append(v["value"])
                batch_units.append(v["unit"])

        batch_values_tensor = torch.tensor(batch_values, dtype=torch.float32, device=self.device).view(-1)
        units_tensor = torch.tensor(batch_units, dtype=torch.float32, device=self.device).long().view(-1)     
        
        batch_units_tensor = self.unit_embed(units_tensor)

        result = torch.cat([batch_values_tensor.unsqueeze(-1), batch_units_tensor], dim=-1)
        return result
    
    def forward(self, input_ids: torch, num_dict: dict):
        trainable = False
        embeddings = self.original_embeddings(input_ids).clone()
        old_embeddings = self.original_embeddings(input_ids).clone()
        batch_dim = embeddings.shape[0]
        inputs_dim = embeddings.shape[1]
        
        embeddings = embeddings.view(-1, self.embedding_dim)

        # Create a mask for the new added tokens
        masked_stocks_fin_embeddings = torch.isin(input_ids, self.stocks_fin_indices_torch)
        masked_num_embeddings = torch.isin(input_ids, self.num_indices_torch)
        
        # Modify the embeddings for the new added tokens only
        if masked_stocks_fin_embeddings.any():
            trainable = True
            new_stocks_fin_input_ids = input_ids[masked_stocks_fin_embeddings]
            selected_ids = self.token_id_to_new_embedding_idx_tensor[new_stocks_fin_input_ids]
            embeddings[masked_stocks_fin_embeddings, :] = self.new_embeddings_layer(selected_ids)
        if masked_num_embeddings.any():
            trainable = True
            new_num_input_ids = input_ids[masked_num_embeddings]
            selected_ids = self.token_id_to_new_embedding_idx_tensor[new_num_input_ids]
            
            embeddings_from_emblayer = self.new_embeddings_layer(selected_ids)

            num_features = self.enhance_num_dict(num_dict)
            num_features = num_features.view(-1, num_features.shape[-1])
            
            embeddings_from_mlp = self.num_mlp(num_features)
            embeddings_from_mlp = embeddings_from_mlp.view(-1, self.embedding_dim)

            embeddings[masked_num_embeddings.view(-1)] = embeddings_from_emblayer + embeddings_from_mlp
            embeddings = embeddings.view(batch_dim, inputs_dim, self.embedding_dim)

        return embeddings, trainable