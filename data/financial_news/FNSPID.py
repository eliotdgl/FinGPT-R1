"""
  We get data from FNSPID, a dataset of financial news 
  https://github.com/Zdong104/FNSPID_Financial_News_Dataset
  https://huggingface.co/datasets/Zihan1004/FNSPID
"""
from datasets import load_dataset

# Load the dataset
data_FNSPID = load_dataset("Zihan1004/FNSPID")
data_FNSPID=data_FNSPID.with_format("pandas")

print(data_FNSPID.head())
