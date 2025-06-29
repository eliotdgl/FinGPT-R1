"""
    == test.py ==
    Model Evaluation

    Tests various sentiment classification models including (if trained):
    - Bert, BertEC
    - HashT, HashTEC
    - DelT, DelTEC
    - FinBERT (optional baseline, no need to train)

    For each model, it evaluates:
    - Accuracy on FinancialPhraseBank-v1.0 (FinP)
    - Accuracy on controlled/generated dataset (GenData)
    - Average confidence
    - Calibration (ECE)
    - Reliability diagram

    Results are stored in:
    - CSV: results/model_comparison.csv
    - Plots: results/plots/<model>/

    Usage:
        python test.py --model <model_names>{['Bert', 'BertEC', 'HashT', 'HashTEC', 'DelT', 'DelTEC', 'all']} [--baseline]
    
    Arguments:
        --model: str -> Required. Name of the model(s) to test.
        --baseline: flag -> Optional. Include FinBERT baseline.
        Example:
            python test.py --model BertEC --baseline
            python test.py --model Bert HashTEC DelT
            python test.py --model all --baseline
"""
import os

os.makedirs("results", exist_ok=True)
os.makedirs("results/plots", exist_ok=True)

import argparse
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

import matplotlib.pyplot as plt 
from sklearn.metrics import accuracy_score
from netcal.metrics import ECE
from netcal.presentation import ReliabilityDiagram

from tokenization.preprocess_text import preprocess_text
from sentiment_analysis.controlled_environment import TextDataGenerator
from sentiment_analysis.sentiment_model_class import Sentiment_Analysis_Model


def evaluate_model(pred_labels_finp, correct_labels_finp, probs_list_finp, pred_labels_gen, correct_labels_gen, probs_list_gen, model_name, df_results) -> None:
    """
        Evaluates a model's predictions on both datasets:
        - Accuracy, average confidence, and ECE
        - Reliability diagrams
    """
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


def get_data():
    """
        Loads or initializes the results DataFrame.
        Loads test dataset (FinP) and Generate controlled (GenData) data.
    """
    df_results = pd.DataFrame(columns=[
        "Accuracy on FinP", "Accuracy on GenData",
        "Avg Confidence on FinP", "Avg Confidence on Gen",
        "ECE on FinP", "ECE on Gen"
    ])

    generator = TextDataGenerator()
    dataset_gen = generator.generate_batch()

    map_num={0:"neutral", 1: "positive", 2: "negative"}

    dataset_finp = pd.read_csv("data/local_data/test_all_agree.csv")

    return df_results, dataset_gen, dataset_finp, map_num


# Baseline (FinBERT)
def finbert_test(df_results, dataset_gen, dataset_finp, map_num) -> None:
    """
        Runs inference and evaluation on FinBERT model on both datasets.
    """
    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

    pred_labels_finp = []
    probs_list_finp = []
    correct_labels_finp = dataset_finp["Label"]
    with torch.no_grad():
        for i, row in tqdm(dataset_finp.iterrows()):
            output = nlp(row["Sentence"])
            probs_list_finp.append([output[0]['score']])
            pred_labels_finp.append(output[0]['label'].lower())

    pred_labels_finp = np.array(pred_labels_finp)
    correct_labels_finp = np.array(correct_labels_finp)

    pred_labels_gen = []
    probs_list_gen = []
    correct_labels_gen = [map_num[num] for num in dataset_gen[1]]
    with torch.no_grad():
        for i, line in tqdm(enumerate(dataset_gen[0])):
            output = nlp(line)
            probs_list_gen.append([output[0]['score']])
            pred_labels_gen.append(output[0]['label'].lower())
    
    pred_labels_gen = np.array(pred_labels_gen)
    correct_labels_gen = np.array(correct_labels_gen)

    evaluate_model(pred_labels_finp, correct_labels_finp, probs_list_finp, pred_labels_gen, correct_labels_gen, probs_list_gen, "FinBERT", df_results)


def test(model: str, df_results, dataset_gen, dataset_finp, map_num, sentiment_model, bert_model: bool = False, special_tokens: bool = False) -> None:
    """
        Evaluates a specified sentiment model on both FinP and GenData datasets.
        Computes accuracy, average confidence and ECE.
        Saves reliability diagrams and updates df_results.
    """
    pred_labels_finp = []
    probs_list_finp = []
    correct_labels_finp = []
    with torch.no_grad():
        for i, row in tqdm(dataset_finp.iterrows()):
            if bert_model:
                processed_text = row["Sentence"]
            else:
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
            if bert_model:
                processed_text = line
            else:
                processed_text, _ = preprocess_text(line) if not special_tokens else (preprocess_text(line, only_special_tokens=True), None)
            probs, pred = sentiment_model.predict(processed_text)
            probs_list_gen.append(probs)
            pred_labels_gen.append(pred)
            correct_labels_gen.append(map_num[dataset_gen[1][i]])

    pred_labels_gen = np.array(pred_labels_gen)
    correct_labels_gen = np.array(correct_labels_gen)

    evaluate_model(pred_labels_finp, correct_labels_finp, probs_list_finp, pred_labels_gen, correct_labels_gen, probs_list_gen, model, df_results)


if __name__ == "__main__":
    try:
        trained_models = [name[:-2] for name in os.listdir("models") if os.path.isdir(os.path.join("models", name))]
    except FileNotFoundError:
        raise ValueError("No trained model found.")
        
    valid_models = trained_models + ['all']
    parser = argparse.ArgumentParser(description="Test Model(s) among: " + ", ".join(valid_models))
    parser.add_argument("--model", 
        nargs="+",
        type=str,
        choices=valid_models,
        required=True,
        help="Model(s) to test. Choose from: " + ", ".join(valid_models))
    parser.add_argument("--baseline", action="store_true", help="Include FinBERT baseline.")
    
    args = parser.parse_args()

    # If 'all' argument to test all, already trained, models at once
    if 'all' in args.model:
        models_to_test = trained_models
    else:
        models_to_test = args.model
    
    df_results, dataset_gen, dataset_finp, map_num = get_data()
    for model in models_to_test:
        print(f"\n==== Testing {model} ====\n")
        sentiment_model = Sentiment_Analysis_Model(load_model=True)
        path = 'models/' + model
        sentiment_model.load(base_path=path)
        special_tokens = "DelT" in model
        bert_model = 'bert' in model.lower()
        test(model, df_results, dataset_gen, dataset_finp, map_num, sentiment_model, bert_model=bert_model, special_tokens=special_tokens)
    
    if args.baseline:
        print("\n==== Testing FinBERT baseline ====")
        finbert_test(df_results, dataset_gen, dataset_finp, map_num)
    
    print("\n\nFinal results:\n\n", df_results)
    df_results.to_csv("results/model_comparison.csv", index=True)
