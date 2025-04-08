"""
We get data from NIFTY, a dataset of financial news 
"""
from datasets import load_dataset

# Load the dataset
def get_data():
    data_NIFTY = load_dataset("raeidsaqur/nifty")
    return data_NIFTY