import os
import json

tokenized_stock_indices = {
    "S&P": "<FinGPTICKER_S&P>",
    "S&P 500": "<FinGPTICKER_S&P500>",
    "S&P500": "<FinGPTICKER_S&P500>",
    "Standard & Poor's 500": "<FinGPTICKER_S&P500>",

    "Dow": "<FinGPTICKER_Dow>",
    "Dow Jones Industrial Average": "<FinGPTICKER_DowJones>",
    "DJIA": "<FinGPTICKER_DowJones>",
    "Dow Jones": "<FinGPTICKER_DowJones>",
    "The Dow": "<FinGPTICKER_DowJones>",

    "Nasdaq": "<FinGPTICKER_NASDAQ>",
    "NASDAQ": "<FinGPTICKER_NASDAQ>",

    "FTSE 100": "<FinGPTICKER_FTSE100>",
    "FTSE100": "<FinGPTICKER_FTSE100>",

    "Nikkei 225": "<FinGPTICKER_Nikkei>",
    "Nikkei": "<FinGPTICKER_Nikkei>",

    "DAX": "<FinGPTICKER_DAX>",

    "CAC 40": "<FinGPTICKER_CAC40>",

    "Hang Seng": "<FinGPTICKER_HSI>",
    "HSI": "<FinGPTICKER_HSI>",

    "Russell 3000": "<FinGPTICKER_Russell>",
    "Russell": "<FinGPTICKER_Russell>",

    "MSCI": "<FinGPTICKER_MSCI>",
    "MSCI World": "<FinGPTICKER_MSCIWorld>",
    "MSCI Emerging Markets": "<FinGPTICKER_MSCIem>"
}


# Export stock indices
if __name__ == "__main__":
    os.makedirs("tokenization/vocabulary", exist_ok=True)
    with open("tokenization/vocabulary/stock_indices_vocab.json", "w") as f:
        json.dump(tokenized_stock_indices, f)
