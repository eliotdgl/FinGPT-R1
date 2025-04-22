import torch
from torch import nn
from tqdm import tqdm
from transformers import LlamaTokenizer, LlamaForCausalLM
import json 

from tokenization.preprocess_text import preprocess_text
from FinGPTR1_tokenizer.custom_embeddings import CustomEmbeddings


def FGPTR1_training(base_model: str = None, base_tokenizer: str = None,
                    data = None,
                    embeddings_path: str = "FinGPTR1_tokenizer_training/saved/custom_embeddings.pt",
                    device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print('\n Importing base tokenizer and model')
    # Load base tokenizer and model
    if base_model is not None and base_tokenizer is not None:
        tokenizer = LlamaTokenizer.from_pretrained(base_model)
        model = LlamaForCausalLM.from_pretrained(base_tokenizer).to(device)
    else:
        tokenizer = LlamaTokenizer.from_pretrained("FinGPTR1_tokenizer/training/base_model/open_llama_3b_model")
        model = LlamaForCausalLM.from_pretrained("FinGPTR1_tokenizer/training/base_model/open_llama_3b_model").to(device)

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

    Custom_Embeddings = CustomEmbeddings(embedding_layer, old_vocab_len, len_vocab_added_stocks_fin, new_vocab_len).to(device)
    optimizer = torch.optim.AdamW(
        list(Custom_Embeddings.new_embeddings_layer.parameters()) +
        list(Custom_Embeddings.num_mlp.parameters()),
        lr=5e-5
    )

    """
    if data is None:
        from data.financial_news.NIFTY import get_data
        data = get_data()
        train_data = data["train"].to_pandas()
        news, labels = train_data["news"], train_data["label"]
        data = news
    """

    dataloader = ["APPL raised by $100"]


    num_epochs = 1
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
        print(f"Training Tokenizer | Epoch {epoch} loss: {epoch_loss}")

    print("Special tokenizer trained successfully")

    torch.save(Custom_Embeddings.state_dict(), embeddings_path)
    print("Tokenizer saved successfully")

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