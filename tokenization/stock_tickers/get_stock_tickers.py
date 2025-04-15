from yahoo_fin import stock_info as si
import yfinance as yf
import pandas as pd
from tqdm import tqdm


# Main stock tickers
df_sp500 = pd.DataFrame(si.tickers_sp500())
#df_nasdaq = pd.DataFrame(si.tickers_nasdaq())
#df_dow = pd.DataFrame(si.tickers_dow())
#df_other = pd.DataFrame(si.tickers_other())

""" Other tickers sources
tickers_ftse100
tickers_ftse250
tickers_ibovespa
tickers_nifty50
tickers_niftybank
"""

# Convert to sets
set_sp500 = set(symbol for symbol in df_sp500[0].values.tolist())
#set_nasdaq = set(symbol for symbol in df_nasdaq[0].values.tolist())
#set_dow = set(symbol for symbol in df_dow[0].values.tolist())
#set_other = set(symbol for symbol in df_other[0].values.tolist())


# Union of all stock tickers
raw_stock_tickers = set.union(set_sp500) #set_nasdaq, set_dow, set_other)
raw_stock_tickers.discard('')

# Clean stock tickers
stock_tickers = set()
remove = ['W', 'R', 'P', 'Q'] # Prefixes of some unwanted stock tickers

for symbol in raw_stock_tickers:
    if len(symbol) > 4 and symbol[-1] in remove:
        continue
    else:
      stock_tickers.add(symbol)

stock_tickers = {symbol for symbol in stock_tickers if not all(c == '-' for c in symbol)}
stock_tickers = {symbol for symbol in stock_tickers if not (symbol[-1] == '$')}
stock_tickers = {symbol for symbol in stock_tickers if len(symbol) >= 2} # So it does not consider start of sentence as tickers

stock_tickers_with_template = {f"<FinGPTICKER_{ticker}>" for ticker in stock_tickers}

# Export stock tickers
import json
import os

if __name__ == "__main__":
    os.makedirs("tokenization/vocabulary", exist_ok=True)
    with open("tokenization/vocabulary/stock_tickers.json", "w") as f:
        json.dump(list(stock_tickers), f)

    os.makedirs("tokenization/vocabulary", exist_ok=True)
    with open("tokenization/vocabulary/stock_tickers_vocab.json", "w") as f:
        json.dump(list(stock_tickers_with_template), f)


"""
# Create dictionary {ticker: company_name}
def get_company_name(ticker):
    try:
        fixed_ticker = ticker.replace("$", "-") # Yahoo Finance usual format 
        stock = yf.Ticker(fixed_ticker)
        return stock.info.get("longName", "NaN")
    except:
        return "NaN"

ticker_name_dict = {}
for ticker in tqdm(stock_tickers, desc="Fetching Company Names"):
    ticker_name_dict[ticker] = get_company_name(ticker)
"""