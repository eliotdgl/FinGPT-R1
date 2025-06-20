import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FinGPTR1_pipeline.FGPTR1_tokenizer import FinGPTR1_Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

os.makedirs("FinGPTR1_pipeline/models", exist_ok=True)

PATH = 'FinGPTR1_pipeline/models/NoMLP'
#PATH = 'FinGPTR1_pipeline/models/NoMLPandGradUnfreeze'
#PATH = 'FinGPTR1_pipeline/models/AllCustom'
#PATH = 'FinGPTR1_pipeline/models/AllCustomandGradUnfreeze'

Fin_tokenizer = FinGPTR1_Tokenizer(PATH, train=True)