import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram

from sentiment_analysis.sentiment_model_class import Sentiment_Analysis_Model
from sentiment_analysis.controlled_environment import TextDataGenerator

from tokenization.preprocess_text import preprocess_text

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

def evaluate_model(pred_labels_finp, correct_labels_finp, probs_list_finp, pred_labels_gen, correct_labels_gen, probs_list_gen, model_name):
    # Accuracy
    accuracy_finp = accuracy_score(correct_labels_finp, pred_labels_finp)
    accuracy_gen = accuracy_score(correct_labels_gen, pred_labels_gen)

    # Average confidence
    avg_conf_finp = np.mean([max(p) for p in probs_list_finp])
    avg_conf_gen = np.mean([max(p) for p in probs_list_gen])

    # Calibration error (ECE)
    ece = ECE(bins=15)
    label_to_index = {"neutral": 0, "positive": 1, "negative": 2}

    correct_numeric_finp = np.array([label_to_index[label] for label in correct_labels_finp])
    probs_np_finp = np.array(probs_list_finp)
    ece_score_finp = ece.measure(probs_np_finp, correct_numeric_finp)
    
    correct_numeric_gen = np.array([label_to_index[label] for label in correct_labels_gen])
    probs_np_gen = np.array(probs_list_gen)
    ece_score_gen = ece.measure(probs_np_gen, correct_numeric_gen)

    rd = ReliabilityDiagram()

    plot_dir = f"results/plots/{model_name}"
    os.makedirs(plot_dir, exist_ok=True)

    plt.figure()
    rd.plot(probs_np_finp, correct_numeric_finp)
    plt.savefig(f"{plot_dir}/finp_reliability_diagram.png")

    plt.figure()
    rd.plot(probs_np_gen, correct_numeric_gen)
    plt.savefig(f"{plot_dir}/gen_reliability_diagram.png")

    plt.close('all')

    df_results.loc[model_name] = [accuracy_finp, accuracy_gen, avg_conf_finp, avg_conf_gen, ece_score_finp, ece_score_gen]
    
    pass


# ===== Inference =====

df_results = pd.DataFrame(columns=[
    "Accuracy on FinP", "Accuracy on GenData",
    "Avg Confidence on FinP", "Avg Confidence on Gen",
    "ECE on FinP", "ECE on Gen"
])

generator = TextDataGenerator()
dataset_gen = generator.generate_batch()

map_num={0:"neutral", 1: "positive", 2: "negative"}

dataset_finp = pd.read_csv("data/local_data/test_all_agree.csv")

# ====================

# Baseline:
model_name = "yiyanghkust/finbert-tone"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

pred_labels_finp = []
probs_list_finp = []
correct_labels_finp = dataset_finp["Label"]
with torch.no_grad():
    for i, row in tqdm(dataset_finp.iterrows()):
        processed_text, _ = preprocess_text(row["Sentence"])
        output = nlp(processed_text)
        probs_list_finp.append([output[0]['score']])
        pred_labels_finp.append(output[0]['label'].lower())

pred_labels_finp = np.array(pred_labels_finp)
correct_labels_finp = np.array(correct_labels_finp)

pred_labels_gen = []
probs_list_gen = []
correct_labels_gen = [map_num[num] for num in dataset_gen[1]]
with torch.no_grad():
    for i, line in tqdm(enumerate(dataset_gen[0])):
        processed_text, _ = preprocess_text(line)
        output = nlp(processed_text)
        probs_list_gen.append([output[0]['score']])
        pred_labels_gen.append(output[0]['label'].lower())
 
pred_labels_gen = np.array(pred_labels_gen)
correct_labels_gen = np.array(correct_labels_gen)

evaluate_model(pred_labels_finp, correct_labels_finp, probs_list_finp, pred_labels_gen, correct_labels_gen, probs_list_gen, "FinBERT")


# ====================


def run_evaluation(model_name, base_path=None, special_tokens=False):
    print(f"\nEvaluating: {model_name}")

    pred_labels_finp = []
    probs_list_finp = []
    correct_labels_finp = []
    with torch.no_grad():
        for i, row in tqdm(dataset_finp.iterrows()):
            processed_text, _ = preprocess_text(row["Sentence"]) if not special_tokens else (preprocess_text(row["Sentence"], only_special_tokens=True), None)
            probs, pred = sentiment_model.predict(processed_text)
            probs_list_finp.append(probs)
            pred_labels_finp.append(pred)
            correct_labels_finp.append(row["Label"])

    pred_labels_finp = np.array(pred_labels_finp)
    correct_labels_finp = np.array(correct_labels_finp)

    pred_labels_gen = []
    probs_list_gen = []
    correct_labels_gen = []
    with torch.no_grad():
        for i, line in tqdm(enumerate(dataset_gen[0])):
            processed_text, _ = preprocess_text(line) if not special_tokens else (preprocess_text(line, only_special_tokens=True), None)
            probs, pred = sentiment_model.predict(processed_text)
            probs_list_gen.append(probs)
            pred_labels_gen.append(pred)
            correct_labels_gen.append(map_num[dataset_gen[1][i]])

    pred_labels_gen = np.array(pred_labels_gen)
    correct_labels_gen = np.array(correct_labels_gen)

    evaluate_model(pred_labels_finp, correct_labels_finp, probs_list_finp, pred_labels_gen, correct_labels_gen, probs_list_gen, model_name)


custom_models = [
    ("BertLoRA", "models/BertLoRA"),
    ("BertLoRAWhole", "models/BertLoRAWhole"),
    #("BertExtLoRA", "models/BertExtLoRA"),
    #("BertExtLoRAWhole", "models/BertExtLoRAWhole"),
    ("NoMLPLoRA", "models/NoMLPLoRA"),
    ("NoMLPLoRAWhole", "models/NoMLPLoRAWhole"),
    ("NoMLPandGradUnfreezeLoRA", "models/NoMLPandGradUnfreezeLoRA"),
    ("NoMLPandGradUnfreezeLoRAWhole", "models/NoMLPandGradUnfreezeLoRAWhole"),
    ("NumLogicLoRA", "models/NumLogicLoRA"),
    ("NumLogicLoRAWhole", "models/NumLogicLoRAWhole")
]


for name, path in custom_models:
    sentiment_model = Sentiment_Analysis_Model(load_model=True)
    sentiment_model.load(base_path=path)
    special_tokens = "NumLogic" in name
    run_evaluation(name, path, special_tokens=special_tokens)
    print(df_results)

print("\n\nFinal results:\n\n", df_results)
df_results.to_csv("results/model_comparison.csv", index=True)