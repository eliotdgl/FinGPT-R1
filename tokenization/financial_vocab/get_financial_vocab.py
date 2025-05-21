financial_vocab = [
    # Market Sentiment - Positive (Bullish)
    #"bullish", "rally", "breakout",
    "uptrend",
    #"gain", "surge", "rebound", "momentum",

    # Market Sentiment - Negative (Bearish)
    #"bearish",
    "crash",
    #"correction", 
    "downtrend", "plunge",
    #"dip", "volatility",

    # Financial Instruments / Assets
    #"equity", "bond", "etf", "futures",
    "crypto", 
    #"commodity",
    "BTC", "bitcoin", "etherum"

    # Economic / Market Indicators
    "inflation",
    #"gdp", "valuation", "liquidity", "recession", "unemployment", "cpi", "pmi", "pct",
    "ppi"

    # Sentiment & Analyst Language
    "outperform",
    #"underperform", "upside", "guidance", "speculation", "sentiment", "confidence", "uncertainty",

    "Q1", "Q2", "Q3", "Q4"
]


# Export financial vocab
import json
import os

if __name__ == "__main__":
    os.makedirs("tokenization/vocabulary", exist_ok=True)
    with open("tokenization/vocabulary/financial_vocab.json", "w") as f:
        json.dump(list(financial_vocab), f)
