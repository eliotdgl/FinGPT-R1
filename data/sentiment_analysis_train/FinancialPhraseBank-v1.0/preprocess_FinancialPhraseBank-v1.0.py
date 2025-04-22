import pandas as pd
import csv
import re
import os 

# Function to process the sentences_allagreed.txt file
def process_sentences(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8',errors='ignore') as infile:
        # Read all lines from the file
        lines = infile.readlines()

    # List to store the rows for CSV output
    csv_data = []

    # Regular expression to split sentence and label
    sentence_label_regex = re.compile(r"(.*?)(@positive|@negative|@neutral)$")

    for line in lines:
        line = line.strip()  # Remove any leading/trailing whitespace
        if line:
            match = sentence_label_regex.match(line)
            if match:
                sentence = match.group(1).strip()
                label = match.group(2)
                label = label.replace('@', '')
                csv_data.append([sentence, label])  # Append the sentence and its label as a row

    # Writing to CSV
    with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
        csv_writer = csv.writer(outfile)
        csv_writer.writerow(['Sentence', 'Label'])  # Writing header
        csv_writer.writerows(csv_data)  # Writing the data

# Usage
output_dir = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0'

input_file = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_AllAgree.txt'  # Input text file
output_file = os.path.join(output_dir, 'Sentences_AllAgree_processed.csv')

process_sentences(input_file, output_file)

print(f"CSV file '{output_file}' has been created successfully.")

input_file = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_50Agree.txt'  # Input text file
output_file = os.path.join(output_dir, 'Sentences_50Agree_processed.csv')

process_sentences(input_file, output_file)

print(f"CSV file '{output_file}' has been created successfully.")

input_file = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_66Agree.txt'  # Input text file
output_file = os.path.join(output_dir, 'Sentences_66Agree_processed.csv')

process_sentences(input_file, output_file)

print(f"CSV file '{output_file}' has been created successfully.")

input_file = 'data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_75Agree.txt'  # Input text file
output_file = os.path.join(output_dir, 'Sentences_75Agree_processed.csv')

process_sentences(input_file, output_file)

print(f"CSV file '{output_file}' has been created successfully.")

df=pd.read_csv('data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_AllAgree_processed.csv')
print(df.head())
