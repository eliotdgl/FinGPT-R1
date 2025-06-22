import os
import sys
import pickle
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentiment_analysis.sentiment_model_class import Sentiment_Analysis_Model
from FinGPTR1_pipeline.training.training_process import FGPTR1_training


# Load the pkl file
print("\nLoading dataset...\n")

with open('data/local_data/DelT_data/generated_data.pkl', 'rb') as f:
    dataset_train = pickle.load(f)
print("\nDataset LOADED\n")


# ===== Training =====

model_name = "bert-base-uncased"

# ====================

print(f"\nTraining: NumLogicLoRA")

INPUT_PATH = "FinGPTR1_pipeline/models/NumLogic"
OUTPUT_PATH = "models/NumLogicLoRA"

# On Financial PhraseBank
FGPTR1_training(INPUT_PATH, model_name, numlogic_model = True)

# On Generated Dataset
sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")

# ====================

print(f"\nTraining: NumLogicLoRAWhole")

OUTPUT_PATH = "models/NumLogicLoRAWhole"

sentiment_model = Sentiment_Analysis_Model(model_name=INPUT_PATH)
print("\nModel LOADED\n")
sentiment_model.train(dataset_train, unfreeze_layers = ['lora_', 'embeddings', 'classifier'])
print("\nModel TRAINED\n")
sentiment_model.save(base_path=OUTPUT_PATH, timestamp_name="1", keep_last=3)

print(f"\nModel saved to: {OUTPUT_PATH}\n")