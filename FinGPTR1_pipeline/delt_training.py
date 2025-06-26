"""
    == delt_training.py ==
    Training script for the Delimiter tokens method.
    
    Performs fine-tuning of a sentiment analysis model 
    using LoRA on both the Financial PhraseBank and generated data 
    Trains and saves two variants:
     - LoRA-only fine-tuned model
     - LoRA + embedding layer + classification head fine-tuned model
"""
import os
import sys
import pickle

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentiment_analysis.sentiment_model_class import Sentiment_Analysis_Model
from FinGPTR1_pipeline.training.training_process import FGPTR1_training


# Load the pkl file generated data
print("\nLoading dataset...\n")
with open('data/local_data/DelT_data/generated_data.pkl', 'rb') as f:
    dataset_train = pickle.load(f)
print("\nDataset LOADED\n")


# ===== Training =====

model_name = "bert-base-uncased" # base model
INPUT_PATH = "FinGPTR1_pipeline/models/NumLogic"

# On Financial PhraseBank
FGPTR1_training(INPUT_PATH, model_name, numlogic_model = True)

# ====================

print(f"\nTraining: NumLogicLoRA")
OUTPUT_PATH = "models/NumLogicLoRA"

# On Generated Dataset
sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_'])
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")

# ====================

print(f"\nTraining: NumLogicLoRAWhole")
OUTPUT_PATH = "models/NumLogicLoRAWhole"

# On Generated Dataset
sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_', 'embeddings', 'classifier'])
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")
