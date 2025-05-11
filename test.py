from FinGPTR1_tokenizer.FGPTR1_tokenizer import FinGPTR1_Tokenizer

tokenizer = FinGPTR1_Tokenizer(train = True)
print("FinGPTR1 tokenizer loaded")

text = ["This is a news"]
print(tokenizer(text))

print("Results printed")