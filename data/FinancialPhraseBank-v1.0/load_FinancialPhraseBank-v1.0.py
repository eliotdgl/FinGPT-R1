"""
    == load_FinancialPhraseBank-v1.0.py ==
    Converts FinancialPhraseBank-v1.0 data, containing labeled financial sentences, into clean CSV files with two columns: Sentence and Label.
    
    Files:
        - Sentences_AllAgree.txt
        - Sentences_50Agree.txt
        - Sentences_66Agree.txt
        - Sentences_75Agree.txt
    
    The output files are saved in the same directory, each suffixed with '_processed.csv'.
"""
import os
import re
import csv
import pandas as pd

def process_sentences(input_file: str, output_file: str) -> None:
    """
        Extracts labeled sentences from a FinancialPhraseBank-v1.0 file and writes them to CSV.
    """
    with open(input_file, 'r', encoding='utf-8',errors='ignore') as infile:
        lines = infile.readlines()

    csv_data = []

    sentence_label_regex = re.compile(r"(.*?)(@positive|@negative|@neutral)$")

    for line in lines:
        line = line.strip()
        if line:
            match = sentence_label_regex.match(line)
            if match:
                sentence = match.group(1).strip()
                label = match.group(2)
                label = label.replace('@', '')
                csv_data.append([sentence, label])

    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Sentence', 'Label'])
        csv_writer.writerows(csv_data)


output_dir = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0'

# AllAgree
input_file = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt'  # Input text file
output_file = os.path.join(output_dir, 'Sentences_AllAgree_processed.csv')

process_sentences(input_file, output_file)
print(f"CSV file '{output_file}' has been created successfully.")

# 75Agree
input_file = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_75Agree.txt'  # Input text file
output_file = os.path.join(output_dir, 'Sentences_75Agree_processed.csv')

process_sentences(input_file, output_file)
print(f"CSV file '{output_file}' has been created successfully.")

# 66Agree
input_file = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_66Agree.txt'  # Input text file
output_file = os.path.join(output_dir, 'Sentences_66Agree_processed.csv')

process_sentences(input_file, output_file)
print(f"CSV file '{output_file}' has been created successfully.")

# 50Agree
input_file = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_50Agree.txt'  # Input text file
output_file = os.path.join(output_dir, 'Sentences_50Agree_processed.csv')

process_sentences(input_file, output_file)
print(f"CSV file '{output_file}' has been created successfully.")
