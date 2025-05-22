import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Sentiment_Analysis.sentiment_model_class import Sentiment_Analysis_Model
from Sentiment_Analysis.controlled_environment import TextDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os
import pickle

def get_train_test_split(data, test_size=0.2, random_state=42):
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("\nInput must be a file path or a DataFrame\n")

    return train_test_split(df, test_size=test_size, random_state=random_state)

"""
Load and preprocessed FinancialPhraseBank-v1.0 all_agree"""


csv_path_all_agree = "data/sentiment_analysis_train/FinancialPhraseBank-v1.0/Sentences_AllAgree_processed.csv"

df_all_agree=pd.read_csv(csv_path_all_agree)

# Split into train and test sets (e.g., 80% train, 20% test)
train_df_all_agree, test_df_all_agree = get_train_test_split(csv_path_all_agree)
train_df_all_agree.to_csv('data/local_data/train_all_agree.csv', index=False)
test_df_all_agree.to_csv('data/local_data/test_all_agree.csv', index=False)
print(train_df_all_agree)

# Initialize and train the model without saving any CSV
sentiment_model = Sentiment_Analysis_Model()
dataset_train_all_agree = sentiment_model.prepare_dataset('data/local_data/train_all_agree.csv') 
print("\n")
for example in dataset_train_all_agree.select(range(2)):
    print(example)
# Save local data
pkl_path_train = os.path.join('data', 'local_data', 'dataset_train_all_agree.pkl')
with open(pkl_path_train, 'wb') as f:
    pickle.dump(dataset_train_all_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'dataset_test_all_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_all_agree, f)



""" Load and preprocessed FinancialPhraseBank-v1.0 75 agree"""

csv_path_75_agree = "data/sentiment_analysis_train/FinancialPhraseBank-v1.0/Sentences_75Agree_processed.csv"

df_75_agree=pd.read_csv(csv_path_75_agree)

# Split into train and test sets (e.g., 80% train, 20% test)
train_df_75_agree, test_df_75_agree = get_train_test_split(csv_path_75_agree)
train_df_75_agree.to_csv('data/local_data/train_75_agree.csv', index=False)
test_df_75_agree.to_csv('data/local_data/test_75_agree.csv', index=False)

# Initialize and train the model without saving any CSV
sentiment_model = Sentiment_Analysis_Model()
dataset_train_75_agree = sentiment_model.prepare_dataset('data/local_data/train_75_agree.csv') 

# Save local data
pkl_path = os.path.join('data', 'local_data', 'dataset_train_75_agree.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(dataset_train_75_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'dataset_train_75_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_75_agree, f)

""" Load and preprocessed FinancialPhraseBank-v1.0 66 agree"""

csv_path_66_agree = "data/sentiment_analysis_train/FinancialPhraseBank-v1.0/Sentences_66Agree_processed.csv"

df_66_agree=pd.read_csv(csv_path_66_agree)

# Split into train and test sets (e.g., 80% train, 20% test)
train_df_66_agree, test_df_66_agree = get_train_test_split(csv_path_66_agree)
train_df_66_agree.to_csv('data/local_data/train_66_agree.csv', index=False)
test_df_66_agree.to_csv('data/local_data/test_66_agree.csv', index=False)

# Initialize and train the model without saving any CSV
sentiment_model = Sentiment_Analysis_Model()
dataset_train_66_agree = sentiment_model.prepare_dataset('data/local_data/train_66_agree.csv') 

# Save local data
pkl_path = os.path.join('data', 'local_data', 'dataset_train_66_agree.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(dataset_train_66_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'dataset_test_66_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_66_agree, f)

""" Load and preprocessed FinancialPhraseBank-v1.0 50 agree"""

csv_path_50_agree = "data/sentiment_analysis_train/FinancialPhraseBank-v1.0/Sentences_50Agree_processed.csv"

train_df_50_agree, test_df_50_agree = get_train_test_split(csv_path_50_agree)
train_df_50_agree.to_csv('data/local_data/train_50_agree.csv', index=False)
test_df_50_agree.to_csv('data/local_data/test_50_agree.csv', index=False)

# Initialize and train the model without saving any CSV
sentiment_model = Sentiment_Analysis_Model()
dataset_train_50_agree = sentiment_model.prepare_dataset('data/local_data/test_50_agree.csv') 

# Save local data
pkl_path = os.path.join('data', 'local_data', 'dataset_train_50_agree.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(dataset_train_50_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'dataset_test_50_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_50_agree, f)

generator=TextDataGenerator()
generated_data=generator.generate_batch()


pkl_path = os.path.join('data', 'local_data', 'generated_data.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(generated_data, f)