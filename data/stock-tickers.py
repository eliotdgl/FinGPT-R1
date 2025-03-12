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

""" Other tickers
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
stock_tickers = set.union(set_sp500, set_nasdaq, set_dow, set_other)
stock_tickers.discard('')


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
