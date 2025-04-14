# General script to preprocess_text
from tokenization.stock_tickers.preprocess_stock_tickers import preprocess_stocks
from tokenization.numericals.preprocess_numericals import Numbers_preprocessor

def preprocess_text(text: str)->str:
    preprocessed_text = preprocess_stocks(text)
    preprocessed_text, numbers_dict = Numbers_preprocessor().preprocess_text(preprocessed_text)
    
    return preprocessed_text, numbers_dict
