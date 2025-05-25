import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
from datasets import load_dataset

from Sentiment_Analysis.sentiment_model_class import Sentiment_Analysis_Model
from Sentiment_Analysis.controlled_environment import TextDataGenerator
from Sentiment_Analysis.data_loader import get_train_test_split

from tokenization.preprocess_text import preprocess_text

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline

from sklearn.metrics import accuracy_score
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram
import matplotlib.pyplot as plt 


def evaluate_model(pred_labels, correct_labels, probs, model_name):
    # Accuracy
    accuracy = accuracy_score(correct_labels, pred_labels)

    # Average confidence
    avg_conf = np.mean([max(p) for p in probs])

    # Calibration error (ECE)
    ece = ECE(bins=15)
    
    label_to_index = {"neutral": 0, "positive": 1, "negative": 2}
    correct_numeric = np.array([label_to_index[label] for label in correct_labels])
    probs_np = np.array(probs)
    ece_score = ece.measure(probs_np, correct_numeric)
    rd = ReliabilityDiagram()

    print(f"\n--Evaluation for: {model_name}--")
    print(f"Accuracy:         {accuracy:.4f}")
    print(f"Avg Confidence:   {avg_conf:.4f}")
    print(f"Calibration (ECE): {ece_score:.4f}")
    rd.plot(probs_np, correct_numeric)
    plt.savefig(f"plots/{model_name}_reliability_diagram.png")
    plt.close()
    
    pass


# ===== Inference =====

generator = TextDataGenerator()
dataset = generator.generate_batch()

map_num={0:"neutral", 1: "positive", 2: "negative"}

# ====================

# Baseline:
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

pred_labels = []
probs_list = []
correct_labels = [map_num[num] for num in dataset[1]]

with torch.no_grad():
    for i, line in tqdm(enumerate(dataset[0])):
        processed_text, _ = preprocess_text(line)
        output = nlp(processed_text)
        probs_list.append([output[0]['score']])
        pred_labels.append(output[0]['label'].lower())
 
pred_labels = np.array(pred_labels)

evaluate_model(pred_labels, correct_labels, probs_list, "FinBERT")


# ====================


def run_evaluation(model_name, base_path=None, special_tokens=False):
    print(f"\nEvaluating: {model_name}")
    pred_labels = []
    probs_list = []
    correct_labels = []

    with torch.no_grad():
        for i, line in tqdm(enumerate(dataset[0])):
            processed_text, _ = preprocess_text(line) if not special_tokens else (preprocess_text(line, only_special_tokens=True), None)
            probs, pred = sentiment_model.predict(processed_text)
            probs_list.append(probs)
            pred_labels.append(pred)
            correct_labels.append(map_num[dataset[1][i]])

    pred_labels = np.array(pred_labels)
    correct_labels = np.array(correct_labels)
    evaluate_model(pred_labels, correct_labels, probs_list, model_name)


custom_models = [
    ("BertLoRA", "models/BertLoRA"),
    ("BertLoRAWhole", "models/BertLoRAWhole"),
    ("NoMLPLoRA", "models/NoMLPLoRA"),
    ("NoMLPLoRAWhole", "models/NoMLPLoRAWhole"),
    ("NoMLPandGradUnfreezeLoRA", "models/NoMLPandGradUnfreezeLoRA"),
    ("NoMLPandGradUnfreezeLoRAWhole", "models/NoMLPandGradUnfreezeLoRAWhole"),
    ("NumLogicLoRA", "models/NumLogicLoRA"),
    ("NumLogicLoRAWhole", "models/NumLogicLoRAWhole"),
]


for name, path in custom_models:
    sentiment_model = Sentiment_Analysis_Model(load_model=True)
    sentiment_model.load(base_path=path)
    special_tokens = "NumLogic" in name
    run_evaluation(name, path, special_tokens=special_tokens)