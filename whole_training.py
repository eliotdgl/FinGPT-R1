import pickle
from sentiment_analysis.sentiment_model_class import Sentiment_Analysis_Model

os.makedirs("models", exist_ok=True)

# Load the pkl file
print("\nLoading dataset...\n")

with open('data/local_data/HashT_data/generated_data.pkl', 'rb') as f:
    dataset_train = pickle.load(f)
print("\nDataset LOADED\n")


# ===== Training =====

train_jobs = [
    {
        "input": None,
        "output": "models/BertLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": None,
        "output": "models/BertLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    },
    #{
    #    "input": "FinGPTR1_pipeline/models/Base",
    #    "output": "models/BertExtLoRA",
    #    "unfreeze": ["lora_"]
    #},
    #{
    #    "input": "FinGPTR1_pipeline/models/Base",
    #    "output": "models/BertExtLoRAWhole",
    #    "unfreeze": ["lora_", "embeddings", "classifier"]
    #},
    {
        "input": "FinGPTR1_pipeline/models/NoMLP",
        "output": "models/NoMLPLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": "FinGPTR1_pipeline/models/NoMLP",
        "output": "models/NoMLPLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    },
    {
        "input": "FinGPTR1_pipeline/models/NoMLPandGradUnfreeze",
        "output": "models/NoMLPandGradUnfreezeLoRA",
        "unfreeze": ["lora_"]
    },
    {
        "input": "FinGPTR1_pipeline/models/NoMLPandGradUnfreeze",
        "output": "models/NoMLPandGradUnfreezeLoRAWhole",
        "unfreeze": ["lora_", "embeddings", "classifier"]
    }
]


for job in train_jobs:
    print(f"\nTraining: {job['output']}")
    sentiment_model = Sentiment_Analysis_Model(model_name=job["input"])
    print(f"Model LOADED")
    sentiment_model.train(dataset_train, unfreeze_layers=job["unfreeze"])
    print(f"Model TRAINED")
    sentiment_model.save(base_path=job["output"], timestamp_name="1", keep_last=3)
    print(f"Model SAVED to: {job['output']}")