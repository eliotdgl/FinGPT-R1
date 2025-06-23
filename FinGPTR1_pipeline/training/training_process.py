import torch
from torch import nn
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from FinGPTR1_pipeline.custom_embeddings import CustomEmbeddings
from tokenization.preprocess_text import preprocess_text


def FGPTR1_training_loop(model, tokenizer, custom_embeddings, unfreeze_schedule, train_loader, val_loader, With_MLP, device, numlogic_model: bool = False):
    
    num_epochs = 40
    epoch_bar = tqdm(range(num_epochs), position=0)

    model.train()
    custom_embeddings.train()
    for epoch in epoch_bar:

        epoch_loss = 0
        batch_bar = tqdm(train_loader, position=1, leave=False)

        if epoch in unfreeze_schedule:
            unfreeze_last_n_layers(model, unfreeze_schedule[epoch])
            if With_MLP:
                optimizer = torch.optim.AdamW(
                list(custom_embeddings.new_embeddings_layer.parameters()) +
                list(custom_embeddings.num_mlp.parameters()) +
                list(custom_embeddings.unit_embed.parameters()) +
                list(filter(lambda p: p.requires_grad, model.parameters())),
                lr=5e-5
                )
            else:
                optimizer = torch.optim.AdamW(
                list(custom_embeddings.new_embeddings_layer.parameters()) +
                list(filter(lambda p: p.requires_grad, model.parameters())),
                lr=5e-5
                )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
            n = unfreeze_schedule[epoch]

        for batch_texts, batch_labels in batch_bar:
            if isinstance(batch_texts, str):
                batch_texts = [batch_texts]

            if numlogic_model:
                preprocessed_batch, batch_numbers_dict = [preprocess_text(text, only_special_tokens=True) for text in batch_texts], None
            else:
                preprocessed_batch, batch_numbers_dict = zip(*[preprocess_text(text) for text in batch_texts])

            inputs = tokenizer(list(preprocessed_batch), padding=True, truncation=True, return_tensors="pt")
            batch_input_ids = inputs["input_ids"].to(device)

            embeddings_batch, trainable = custom_embeddings(batch_input_ids, batch_numbers_dict)

            if not trainable:
                continue
            
            labels = torch.tensor(batch_labels).to(device)

            output = model(inputs_embeds=embeddings_batch, labels=labels)
            
            optimizer.zero_grad()
            loss = output.loss
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        
        scheduler.step()
        val_loss = evaluate(model, custom_embeddings, val_loader, tokenizer, device)
        print(f"\nTraining Tokenizer with the {n}th layers unfrozen | Epoch {epoch+1} training loss: {epoch_loss} / val loss: {val_loss}")



def FGPTR1_training(PATH: str,
                    base_model: str = "bert-base-uncased",
                    With_MLP: bool = False,
                    device = None,
                    numlogic_model = False):
    """
    This function is the entry point for training the FinGPTR1 tokenizer.
    It sets up the device and calls the training loop.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('\nImporting base tokenizer and model\n')
    
    # Load base tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    config = AutoConfig.from_pretrained(base_model, num_labels=3)
    model = AutoModelForSequenceClassification.from_pretrained(base_model, config=config).to(device)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    old_vocab_len = len(tokenizer)
    print(old_vocab_len)

    # Import new tokens
        # Load stock indices vocabulary
    with open("tokenization/vocabulary/stock_indices_vocab.json", "r") as f:
        stock_indices = list(json.load(f).values())
        # Load stock tickers vocabulary
    with open("tokenization/vocabulary/stock_tickers_vocab.json", "r") as f:
        stock_tickers = json.load(f)
        # Load financial vocabulary
    with open("tokenization/vocabulary/financial_vocab.json", "r") as f:
        financial_tokens = json.load(f)
    if not numlogic_model:
            # Load numerical vocabulary
        with open("tokenization/vocabulary/numericals_vocab.json", "r") as f:
            num_tokens = json.load(f)

    
    def already_in_vocab(tokenizer, tokens):
        existing_vocab = set(tokenizer.get_vocab().keys())
        for token in tokens:
            if token in existing_vocab:
                print("Token already in tokenizer's vocab: ", token)
            else:
                tokenizer.add_tokens(token)

    # Add new tokens to the tokenizer
    if numlogic_model:
        special_tokens = ["<SON>", "<VAL>", "<EON>"]
        tokenizer.add_special_tokens({"additional_special_tokens": special_tokens})
        already_in_vocab(tokenizer, stock_indices)
        already_in_vocab(tokenizer, stock_tickers)
        already_in_vocab(tokenizer, financial_tokens)

        len_vocab_added_stocks_fin = len(tokenizer)
        new_vocab_len = len(tokenizer)
    else:        
        already_in_vocab(tokenizer, stock_indices)
        already_in_vocab(tokenizer, stock_tickers)
        already_in_vocab(tokenizer, financial_tokens)

        len_vocab_added_stocks_fin = len(tokenizer)

        already_in_vocab(tokenizer, num_tokens)

        new_vocab_len = len(tokenizer)

    print(len_vocab_added_stocks_fin)
    print(new_vocab_len)
    # Resize the model's embedding layer to accommodate the new added tokens
    model.resize_token_embeddings(new_vocab_len)

    # Get embedding layer of Llama model
    embedding_layer = model.get_input_embeddings()

    Custom_Embeddings = CustomEmbeddings(embedding_layer, old_vocab_len, len_vocab_added_stocks_fin, new_vocab_len, With_MLP, device).to(device)

    #from datasets import load_from_disk
    #data = load_from_disk("data/nifty_dataset_local")
    #train_data = data["train"].to_pandas()
    #val_data = data["valid"].to_pandas()
    
    #train_data = train_data[train_data["news"].str.len() <= 500]
    #val_data = val_data[val_data["news"].str.len() <= 500]

    #train_news = train_data["news"].tolist()
    #train_labels = train_data["label"].tolist()

    #val_news = val_data["news"].tolist()
    #val_labels = val_data["label"].tolist()

    #label_map = {"Neutral": 0, "Rise": 1, "Fall": 2}  # Map sentiment classes
    from datasets import load_dataset
    data = load_dataset('csv', data_files='data/local_data/train_all_agree.csv')
    from sklearn.model_selection import train_test_split
    train_news, val_news, train_labels, val_labels = train_test_split(data["train"]["Sentence"], data["train"]["Label"], test_size=0.1)

    label_map = {"neutral": 0, "positive": 1, "negative": 2}  # Map sentiment classes
    #labels = [label_map[label] for label in labels]
    train_labels = [label_map[label] for label in train_labels]
    val_labels = [label_map[label] for label in val_labels]
    

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False


    if 'GradUnfreeze' in PATH:
        unfreeze_schedule = {0: 0, 10: 1, 20: 3, 30: 5}
    else:
        unfreeze_schedule = {0: 0}

    train_loader = DataLoader(list(zip(train_news, train_labels)), batch_size=32, shuffle=True)
    val_loader = DataLoader(list(zip(val_news, val_labels)), batch_size=32, shuffle=False)

    FGPTR1_training_loop(model, tokenizer, Custom_Embeddings, unfreeze_schedule, train_loader, val_loader, With_MLP, device, numlogic_model)

    print("\nSpecial tokenizer trained successfully\n")

    os.makedirs(PATH + "/tokenizer", exist_ok=True)
    os.makedirs(PATH + "/model", exist_ok=True)
    os.makedirs(PATH + "/custom_embeddings", exist_ok=True)

    with torch.no_grad():
        # Update the model's embedding layer
        embedding_layer.weight[old_vocab_len:] = Custom_Embeddings.new_embeddings_layer.weight.clone()
    
    tokenizer.save_pretrained(PATH + '/tokenizer')
    model.save_pretrained(PATH + '/model')

    torch.save(Custom_Embeddings.state_dict(), PATH + "/custom_embeddings/custom_embeddings.pt")

    metadata = {
        "old_vocab_len": old_vocab_len,
        "len_vocab_added_stocks_fin": len_vocab_added_stocks_fin,
        "new_vocab_len": new_vocab_len
    }
    with open(PATH + "/custom_embeddings/custom_embeddings_meta.json", "w") as f:
        json.dump(metadata, f)

    print(f"\nFinGPTR1 Model saved successfully: {PATH}\n")


def unfreeze_last_n_layers(model, n=0):
    if n == 0:
        pass
    transformer_layers = model.bert.encoder.layer
    for layer in transformer_layers[-n:]:
        for param in layer.parameters():
            param.requires_grad = True
    print(f"\n-- Unfreezing last {n} layer(s) of the model --\n")


def evaluate(model, custom_embeddings, dataloader, tokenizer, device):
    model.eval()
    custom_embeddings.eval()
    total_loss = 0

    with torch.no_grad():
        for batch_texts, batch_labels in dataloader:
            if isinstance(batch_texts, str):
                batch_texts = [batch_texts]
            
            preprocessed_batch, batch_numbers_dict = zip(*[preprocess_text(text) for text in batch_texts])

            inputs = tokenizer(list(preprocessed_batch), padding=True, truncation=True, return_tensors="pt")
            batch_input_ids = inputs["input_ids"].to(device)

            embeddings_batch, _ = custom_embeddings(batch_input_ids, batch_numbers_dict)
            
            labels = torch.tensor(batch_labels).to(device)
            output = model(inputs_embeds=embeddings_batch, labels=labels)
            
            loss = output.loss

            total_loss += loss.item()
    
    model.train()
    custom_embeddings.train()
    return total_loss / len(dataloader)