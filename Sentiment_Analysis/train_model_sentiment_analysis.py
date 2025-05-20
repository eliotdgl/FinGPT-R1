from sentiment_model_class import Sentiment_Analysis_Model
from controlled_environment import TextDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os
from data_loader import get_train_test_split
import pickle
# Load the CSV file
with open('data/local_data/dataset_train_all_agree.pkl', 'rb') as f:
    dataset_train_all_agree = pickle.load(f)
sentiment_model = Sentiment_Analysis_Model()
sentiment_model.train(dataset_train_all_agree)
sentiment_model.save(base_path="./sentiment_model_test_1", timestamp_name="test_1", keep_last=3)


