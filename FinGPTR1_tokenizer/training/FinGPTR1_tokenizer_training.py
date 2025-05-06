import torch
from torch import nn
import os
import json
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM
import json 

from tokenization.preprocess_text import preprocess_text
from FinGPTR1_tokenizer.custom_embeddings import CustomEmbeddings
from torch.utils.data import DataLoader


def FGPTR1_training(base_model: str = None,
                    embeddings_path: str = "FinGPTR1_tokenizer/saved/custom_embeddings.pt",
                    device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('\nImporting base tokenizer and model')
    # Load base tokenizer and model
    if base_model is not None:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(base_model).to(device)
    else:
        tokenizer = LlamaTokenizer.from_pretrained("openlm-research/open_llama_3b")
        model = LlamaForCausalLM.from_pretrained("openlm-research/open_llama_3b").to(device)


    old_vocab_len = len(tokenizer)

    # Import new tokens
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

    """
    def already_in_vocab(tokenizer, tokens):
        existing_vocab = set(tokenizer.get_vocab().keys())
        for token in tokens:
            if token in existing_vocab:
                print("Token already in tokenizer's vocab: ", token)

    already_in_vocab(tokenizer, stock_indices)
    already_in_vocab(tokenizer, stock_tickers)
    already_in_vocab(tokenizer, num_tokens)
    already_in_vocab(tokenizer, financial_tokens)
    """
    
    # Add new tokens to the tokenizer
    tokenizer.add_tokens(stock_indices)
    tokenizer.add_tokens(stock_tickers)
    tokenizer.add_tokens(financial_tokens)

    len_vocab_added_stocks_fin = len(tokenizer)

    tokenizer.add_tokens(num_tokens)

    tokenizer.pad_token = tokenizer.eos_token
    
    new_vocab_len = len(tokenizer)


    # Resize the model's embedding layer to accommodate the new added tokens
    model.resize_token_embeddings(new_vocab_len)

    # Freeze all original embeddings from Llama tokenizer to prevent them from being updated during training
    for param in model.parameters():
        param.requires_grad = False


    # Get embedding layer of Llama model
    embedding_layer = model.get_input_embeddings()

    Custom_Embeddings = CustomEmbeddings(embedding_layer, old_vocab_len, len_vocab_added_stocks_fin, new_vocab_len).to(device)
    optimizer = torch.optim.AdamW(
        list(Custom_Embeddings.new_embeddings_layer.parameters()) +
        list(Custom_Embeddings.num_mlp.parameters()) +
        list(Custom_Embeddings.unit_embed.parameters()),
        lr=5e-5
    )


    from datasets import load_from_disk
    data = load_from_disk("data/nifty_dataset_local")

    """
    from data.financial_news.NIFTY import get_data
    data = get_data()
    """
    train_data = data["train"].to_pandas()
    news = [headline for entry in train_data["news"].tolist() for headline in entry.split('\n')]
    #labels = [label for entry in train_data["label"].tolist() for label in entry.split('\n')]
    dataloader = DataLoader(news, batch_size=16, shuffle=True)
    
    num_epochs = 1
    epoch_bar = tqdm(range(num_epochs), position=0)

    # Training loop to train EmbeddingMLP() 
    Custom_Embeddings.train()
    for epoch in epoch_bar:

        epoch_loss = 0
        batch_bar = tqdm(dataloader, position=1, leave=False)
        for batch in batch_bar:
            if isinstance(batch, str):
                batch = [batch]
            preprocessed_batch, batch_numbers_dict = zip(*[preprocess_text(text) for text in batch])

            inputs = tokenizer(list(preprocessed_batch), padding=True, truncation=True, return_tensors="pt")
            batch_input_ids = inputs["input_ids"].to(device)

            embeddings_batch, trainable = Custom_Embeddings(batch_input_ids, batch_numbers_dict)

            if not trainable:
                continue

            output = model(inputs_embeds=embeddings_batch, labels=batch_input_ids)
            
            optimizer.zero_grad()
            loss = output.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        print(f"Training Tokenizer | Epoch {epoch} loss: {epoch_loss}")

    print("Special tokenizer trained successfully")

    os.makedirs("FinGPTR1_tokenizer/saved", exist_ok=True)

    torch.save(Custom_Embeddings.state_dict(), embeddings_path)
    print("Tokenizer saved successfully")

    metadata = {
        "old_vocab_len": old_vocab_len,
        "len_vocab_added_stocks_fin": len_vocab_added_stocks_fin,
        "new_vocab_len": new_vocab_len
    }
    with open("FinGPTR1_tokenizer/saved/custom_embeddings_meta.json", "w") as f:
        json.dump(metadata, f)
    
    """
    with torch.no_grad():
        # Update the model's embedding layer
        embedding_layer.weight[old_vocab_len:] = Custom_Embeddings.new_embeddings_layer.weight.clone()

    # Save the personalized tokenizer
    tokenizer.save_pretrained(tokenizer_path)
    model.save_pretrained(tokenizer_path)
    torch.save(Custom_Embeddings.state_dict(), embeddings_path)

    print("Tokenizer saved successfully")
    """