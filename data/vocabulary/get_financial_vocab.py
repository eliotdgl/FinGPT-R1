financial_vocab = [
    # Market Sentiment - Positive (Bullish)
    "bullish", "rally", "breakout", "uptrend",
    "gain", "surge", "rebound", "momentum",

    # Market Sentiment - Negative (Bearish)
    "bearish", "crash", "correction", "downtrend",
    "loss", "plunge", "dip", "volatility",

    # Financial Instruments / Assets
    "stock", "equity", "bond", "etf", "option", "futures",
    "crypto", "commodity", "index",

    # Economic / Market Indicators
    "inflation", "gdp",
    "valuation", "liquidity",
    "recession", "unemployment", "cpi", "pmi",

    # Sentiment & Analyst Language
    "outperform", "underperform",
    "upside", "risk", "guidance", "speculation", "sentiment",
    "confidence", "uncertainty"
]


# Export financial vocab
import json
import os

if __name__ == "__main__":
    os.makedirs("data/vocabulary", exist_ok=True)
    with open("data/vocabulary/financial_vocab.json", "w") as f:
        json.dump(list(financial_vocab), f)