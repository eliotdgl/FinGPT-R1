import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FinGPTR1_pipeline.FGPTR1_tokenizer import FinGPTR1_Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

os.makedirs("FinGPTR1_pipeline/models", exist_ok=True)

PATHS = ['FinGPTR1_pipeline/models/HashT', 'FinGPTR1_pipeline/models/HashTandGradUnfreeze', 'FinGPTR1_pipeline/models/AllCustom', 'FinGPTR1_pipeline/models/AllCustomandGradUnfreeze']

#PATH = 'FinGPTR1_pipeline/models/HashT'
#PATH = 'FinGPTR1_pipeline/models/HashTandGradUnfreeze'
#PATH = 'FinGPTR1_pipeline/models/AllCustom'
#PATH = 'FinGPTR1_pipeline/models/AllCustomandGradUnfreeze'

for PATH in PATHS:
    Fin_tokenizer = FinGPTR1_Tokenizer(PATH, train=True)