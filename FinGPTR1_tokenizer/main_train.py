from FGPTR1_tokenizer import FinGPTR1_Tokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

base_model= "yiyanghkust/finbert-tone"

tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForSequenceClassification.from_pretrained(base_model)


#PATH = 'FinGPTR1_tokenizer/models/Base'
PATH = 'FinGPTR1_tokenizer/models/NoMLP'
#PATH = 'FinGPTR1_tokenizer/models/NoMLPandGradUnfreeze'
#PATH = 'FinGPTR1_tokenizer/models/AllCustom'
#PATH = 'FinGPTR1_tokenizer/models/AllCustomandGradUnfreeze'

Fin_tokenizer = FinGPTR1_Tokenizer(PATH, train = True)
print("FinGPTR1 tokenizer loaded")

text = ["This is a news with +$10M in stock"]

model.eval()

# Tokenize & predict
inputs = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs, dim=-1).item()
print(probs, predicted_class)

print('Tokenize: ', Fin_tokenizer.tokenize(text))
print('Embeddings: ', Fin_tokenizer.get_embeddings(text))
print('Predict: ', Fin_tokenizer(text))

print("Results printed")