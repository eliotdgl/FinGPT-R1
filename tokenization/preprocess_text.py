# General script to preprocess_text
from tokenization.stock_tickers.preprocess_stock_tickers import preprocess_stocks
from tokenization.numericals.preprocess_numericals import preproccess_numbers

def preprocess_text(text: str)->str:
    preprocessed_text = preproccess_numbers(text)
    preprocessed_text = preprocess_stocks(preprocessed_text)

    return preprocessed_text
