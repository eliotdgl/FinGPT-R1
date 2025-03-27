!pip install requests-html yahoo_fin yfinance tqdm

from yahoo_fin import stock_info as si
import yfinance as yf
import pandas as pd
from tqdm import tqdm


# Main stock tickers
df_sp500 = pd.DataFrame(si.tickers_sp500())
df_nasdaq = pd.DataFrame(si.tickers_nasdaq())
df_dow = pd.DataFrame(si.tickers_dow())
df_other = pd.DataFrame(si.tickers_other())

""" Other tickers sources
tickers_ftse100
tickers_ftse250
tickers_ibovespa
tickers_nifty50
tickers_niftybank
"""

# Convert to sets
set_sp500 = set(symbol for symbol in df_sp500[0].values.tolist())
set_nasdaq = set(symbol for symbol in df_nasdaq[0].values.tolist())
set_dow = set(symbol for symbol in df_dow[0].values.tolist())
set_other = set(symbol for symbol in df_other[0].values.tolist())


# Union of all stock tickers
raw_stock_tickers = set.union(set_sp500, set_nasdaq, set_dow, set_other)
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


tokenized_stock_indices = {
    "S&P 500": "<FinGPTICKER_S&P500>",
    "S&P500": "<FinGPTICKER_S&P500>",
    "Standard & Poor's 500": "<FinGPTICKER_S&P500>",

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

    "MSCI": "<FinGPTICKER_MSCI>"
}


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
