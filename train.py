import os

os.makedirs("FinGPTR1_pipeline/models", exist_ok=True)
os.makedirs("models", exist_ok=True)

import sys
import pickle
import argparse
from FinGPTR1_pipeline.FGPTR1_tokenizer import FinGPTR1_Tokenizer
from FinGPTR1_pipeline.training.training_process import FGPTR1_training
from BERT.bert_training import bert_train, bertec_train
from sentiment_analysis.sentiment_model_class import Sentiment_Analysis_Model

def train(model: str, dataset_train, dataset_train_hasht, dataset_train_delt):

    if model in ['HashT', 'HashTEC']:
        PATH = 'FinGPTR1_pipeline/models/HashT'
        Fin_tokenizer = FinGPTR1_Tokenizer(PATH, train=True)
        sentiment_model = Sentiment_Analysis_Model(model_name=PATH)
        if model == 'HashT':
            sentiment_model.train(dataset_train_hasht, unfreeze_layers=["lora_"])
            sentiment_model.save(base_path='models/HashTLoRA', timestamp_name="1", keep_last=3)
        else:
            sentiment_model.train(dataset_train_hasht, unfreeze_layers=["lora_", "embeddings", "classifier"])
            sentiment_model.save(base_path='models/HashTLoRAEC', timestamp_name="1", keep_last=3)

    elif model in ['DelT', 'DelTEC']:
        PATH = 'FinGPTR1_pipeline/models/DelT'
        FGPTR1_training(PATH, numlogic_model = True)
        sentiment_model = Sentiment_Analysis_Model(model_name=PATH)
        if model == 'DelT':
            sentiment_model.train(dataset_train_delt, unfreeze_layers = ['lora_'])
            sentiment_model.save(base_path='models/DelTLoRA', timestamp_name="1", keep_last=3)
        else:
            sentiment_model.train(dataset_train_delt, unfreeze_layers = ["lora_", "embeddings", "classifier"])
            sentiment_model.save(base_path='models/DelTLoRAEC', timestamp_name="1", keep_last=3)
    
    elif model in ['BERT', 'BERTEC']:
        os.makedirs("BERT/models", exist_ok=True)
        sentiment_model = Sentiment_Analysis_Model(load_model=True)
        if model == 'BERT':
            bert_train()
            sentiment_model.load(base_path="BERT/models/BertLoRA")
            sentiment_model.train(dataset_train, unfreeze_layers=["lora_"])
            sentiment_model.save(base_path="models/BertLoRA", timestamp_name="1", keep_last=3)
        else:
            bertec_train()
            sentiment_model.load(base_path="BERT/models/BertLoRAWhole")
            sentiment_model.train(dataset_train, unfreeze_layers=["lora_", "embeddings", "classifier"])
            sentiment_model.save(base_path="models/BertLoRAEC", timestamp_name="1", keep_last=3)
    else:
        raise ValueError(f'Model {model} not recognized')

    pass


if __name__ == "__main__":
    valid_models = ['BERT', 'BERTEC', 'HashT', 'HashTEC', 'DelT', 'DelTEC', 'all']
    parser = argparse.ArgumentParser(description="Train Model")
    parser.add_argument("--model", 
        nargs="+",
        type=str,
        choices=valid_models,
        required=True,
        help="Model(s) to train. Choose from: " + ", ".join(valid_models))
    
    args = parser.parse_args()
    
    if 'all' in args.model:
        models_to_train = ['BERT', 'BERTEC', 'HashT', 'HashTEC', 'DelT', 'DelTEC']
    else:
        models_to_train = args.model

    with open('data/local_data/generated_data.pkl', 'rb') as f:
        dataset_train = pickle.load(f)
    with open('data/local_data/HashT_data/generated_data.pkl', 'rb') as f:
        dataset_train_hasht = pickle.load(f)
    with open('data/local_data/DelT_data/generated_data.pkl', 'rb') as f:
        dataset_train_delt = pickle.load(f)

    for model in models_to_train:
        print(f"\n==== Training {model} ====\n")
        train(model, dataset_train, dataset_train_hasht, dataset_train_delt)
