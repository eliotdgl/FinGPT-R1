"""
    == preprocess_text.py ==
    General script to preprocess text.
    Unified `preprocess_text` function that applies:
    - Stock ticker preprocessing
    - Numerical token preprocessing
"""
from tokenization.stock_tickers.preprocess_stock_tickers import preprocess_stocks
from tokenization.numericals.preprocess_numericals import Numbers_preprocessor

def preprocess_text(text: str, only_special_tokens: bool = False) -> str:
    if only_special_tokens:
        preprocessed_text, _ = Numbers_preprocessor().preprocess_text(text, only_special_tokens)
        return preprocessed_text
    else:
        preprocessed_text = preprocess_stocks(text)
        preprocessed_text, numbers_dict = Numbers_preprocessor().preprocess_text(preprocessed_text)
        return preprocessed_text, numbers_dict
