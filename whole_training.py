import os
import pickle
from sentiment_analysis.sentiment_model_class import Sentiment_Analysis_Model

os.makedirs("models", exist_ok=True)

# Load the pkl file
print("\nLoading dataset...\n")
with open('data/local_data/generated_data.pkl', 'rb') as f:
    dataset_train = pickle.load(f)
with open('data/local_data/HashT_data/generated_data.pkl', 'rb') as f:
    dataset_train_hasht = pickle.load(f)
print("\nDataset LOADED\n")


# ===== Training =====

bert_train = [
    {
        "input": "BERT/models/BertLoRA",
        "output": "models/BertLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": "BERT/models/BertLoRAWhole",
        "output": "models/BertLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    }
]

train_jobs = [
    {
        "input": "FinGPTR1_pipeline/models/HashT",
        "output": "models/HashTLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": "FinGPTR1_pipeline/models/HashT",
        "output": "models/HashTLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    },
    {
        "input": "FinGPTR1_pipeline/models/HashTandGradUnfreeze",
        "output": "models/HashTandGradUnfreezeLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": "FinGPTR1_pipeline/models/HashTandGradUnfreeze",
        "output": "models/HashTandGradUnfreezeLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    }
]

for bert_model in bert_train:
    sentiment_model = Sentiment_Analysis_Model(load_model=True)
    sentiment_model.load(base_path=bert_model["input"])
    sentiment_model.train(dataset_train, unfreeze_layers=bert_model["unfreeze"])
    sentiment_model.save(base_path=bert_model["output"], timestamp_name="1", keep_last=3)


for job in train_jobs:
    print(f"\nTraining: {job['output']}")
    sentiment_model = Sentiment_Analysis_Model(model_name=job["input"])
    print(f"Model LOADED")
    sentiment_model.train(dataset_train_hasht, unfreeze_layers=job["unfreeze"])
    print(f"Model TRAINED")
    sentiment_model.save(base_path=job["output"], timestamp_name="1", keep_last=3)
    print(f"Model SAVED to: {job['output']}")