"""
    == local_data_loader.py ==
    Prepares and saves training/test datasets from FinancialPhraseBank-v1.0 and Generated data
    
    Note:
    - Output data is saved under:
        data/local_data/
        ├── DelT_data/
        ├── HashT_data/
        └── (base path for AllAgree/75/66/50 splits)
"""
import os
import sys
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sentiment_analysis.sentiment_model_class import Sentiment_Analysis_Model
from sentiment_analysis.controlled_environment import TextDataGenerator

os.makedirs("data/local_data", exist_ok=True)
os.makedirs("data/local_data/DelT_data", exist_ok=True)
os.makedirs("data/local_data/HashT_data", exist_ok=True)

sentiment_model = Sentiment_Analysis_Model(load_model=True)

def get_train_test_split(data, test_size: float = 0.2, random_state: int = 42):
    if isinstance(data, str):
        df = pd.read_csv(data)
    elif isinstance(data, pd.DataFrame):
        df = data
    else:
        raise ValueError("\nInput must be a file path or a DataFrame\n")

    return train_test_split(df, test_size=test_size, random_state=random_state)


"""Load and preprocessed FinancialPhraseBank-v1.0 all_agree"""
csv_path_all_agree = "data/FinancialPhraseBank-v1.0/Sentences_AllAgree_processed.csv"

df_all_agree=pd.read_csv(csv_path_all_agree)

# Split into train and test sets (e.g., 80% train, 20% test)
train_df_all_agree, test_df_all_agree = get_train_test_split(csv_path_all_agree)
train_df_all_agree.to_csv('data/local_data/train_all_agree.csv', index=False)
test_df_all_agree.to_csv('data/local_data/test_all_agree.csv', index=False)

# Initialize and train the model without saving any CSV
    # DelT
dataset_train_all_agree = sentiment_model.prepare_dataset('data/local_data/train_all_agree.csv', numlogic_model = True) 

# Save local data
pkl_path_train = os.path.join('data', 'local_data', 'DelT_data', 'dataset_train_all_agree.pkl')
with open(pkl_path_train, 'wb') as f:
    pickle.dump(dataset_train_all_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'DelT_data', 'dataset_test_all_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_all_agree, f)

    # HashT
dataset_train_all_agree = sentiment_model.prepare_dataset('data/local_data/train_all_agree.csv') 

# Save local data
pkl_path_train = os.path.join('data', 'local_data', 'HashT_data', 'dataset_train_all_agree.pkl')
with open(pkl_path_train, 'wb') as f:
    pickle.dump(dataset_train_all_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'HashT_data', 'dataset_test_all_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_all_agree, f)


""" Load and preprocessed FinancialPhraseBank-v1.0 75 agree"""
csv_path_75_agree = "data/FinancialPhraseBank-v1.0/Sentences_75Agree_processed.csv"

df_75_agree=pd.read_csv(csv_path_75_agree)

# Split into train and test sets (e.g., 80% train, 20% test)
train_df_75_agree, test_df_75_agree = get_train_test_split(csv_path_75_agree)
train_df_75_agree.to_csv('data/local_data/train_75_agree.csv', index=False)
test_df_75_agree.to_csv('data/local_data/test_75_agree.csv', index=False)

# Initialize and train the model without saving any CSV
dataset_train_75_agree = sentiment_model.prepare_dataset('data/local_data/train_75_agree.csv') 

# Save local data
pkl_path = os.path.join('data', 'local_data', 'dataset_train_75_agree.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(dataset_train_75_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'dataset_train_75_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_75_agree, f)


""" Load and preprocessed FinancialPhraseBank-v1.0 66 agree"""
csv_path_66_agree = "data/FinancialPhraseBank-v1.0/Sentences_66Agree_processed.csv"

df_66_agree=pd.read_csv(csv_path_66_agree)

# Split into train and test sets (e.g., 80% train, 20% test)
train_df_66_agree, test_df_66_agree = get_train_test_split(csv_path_66_agree)
train_df_66_agree.to_csv('data/local_data/train_66_agree.csv', index=False)
test_df_66_agree.to_csv('data/local_data/test_66_agree.csv', index=False)

# Initialize and train the model without saving any CSV
dataset_train_66_agree = sentiment_model.prepare_dataset('data/local_data/train_66_agree.csv') 

# Save local data
pkl_path = os.path.join('data', 'local_data', 'dataset_train_66_agree.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(dataset_train_66_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'dataset_test_66_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_66_agree, f)


""" Load and preprocessed FinancialPhraseBank-v1.0 50 agree"""
csv_path_50_agree = "data/FinancialPhraseBank-v1.0/Sentences_50Agree_processed.csv"

train_df_50_agree, test_df_50_agree = get_train_test_split(csv_path_50_agree)
train_df_50_agree.to_csv('data/local_data/train_50_agree.csv', index=False)
test_df_50_agree.to_csv('data/local_data/test_50_agree.csv', index=False)

dataset_train_50_agree = sentiment_model.prepare_dataset('data/local_data/test_50_agree.csv') 

# Save local data
pkl_path = os.path.join('data', 'local_data', 'dataset_train_50_agree.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(dataset_train_50_agree, f)

pkl_path_test = os.path.join('data', 'local_data', 'dataset_test_50_agree.pkl')
with open(pkl_path_test, 'wb') as f:
    pickle.dump(test_df_50_agree, f)


# -- Generated Data --
generator=TextDataGenerator(number_of_episode=5)
generated_data=generator.generate_batch()

gen = sentiment_model.prepare_dataset(generated_data, preprocess = False, numlogic_model = True)

pkl_path = os.path.join('data', 'local_data', 'generated_data.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(gen, f)

    # DelT
gen_delt = sentiment_model.prepare_dataset(generated_data, numlogic_model = True) 

pkl_path = os.path.join('data', 'local_data', 'DelT_data', 'generated_data.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(gen_delt, f)

    # HashT
gen_hasht = sentiment_model.prepare_dataset(generated_data) 

pkl_path = os.path.join('data', 'local_data', 'HashT_data', 'generated_data.pkl')
with open(pkl_path, 'wb') as f:
    pickle.dump(gen_hasht, f)
