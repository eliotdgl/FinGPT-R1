from transformers import AutoTokenizer
import re

stock_tickers = {} # Import stock tickers from /data/stock-tickers.py
stock_tickers_w_dollar = stock_tickers.union({'$'+symbol for symbol in stock_tickers})

def preprocess_stock_tickers(token: str) -> str:
  """
    Identify stock tickers in text and convert them with special template
  """
  regex_pattern = r'\$?\b[A-Z]+(?:[\.|\$][A-Za-z0-9]+)*\b'
  potential_tickers = re.findall(regex_pattern, token)

  def replace_ticker(match):
    ticker = match.group(0)
    if ticker.startswith("$"):
        ticker = ticker[1:]

    return f"<FinGPTICKER_{ticker}>" if ticker in stock_tickers_w_dollar else ticker

  processed_text = re.sub(regex_pattern, replace_ticker, token)

  return processed_text


def reverse_preprocess_stock_ticker(token: str, space_marker: str = None) -> str:
  """
    Convert back special template to stock tickers
  """
  if token.startswith(space_marker + "<FinGPTICKER") and token.endswith(">"):
    return token[0] + token[14:-1]
  return token


def merge_space_marker(tokens: list[str], space_marker: str = None) -> list[str]:
  """
    Merge tokenized stock tickers and tokenizer's space marker
  """
  merged_tokens = []
  for i, token in enumerate(tokens):
    if token == space_marker and i + 1 < len(tokens) and tokens[i+1].startswith("<FinGPTICKER_"):
      merged_tokens.append(space_marker + tokens[i+1])
    elif not token.startswith("<FinGPTICKER_"):
      merged_tokens.append(token)

  return merged_tokens



base_model = 'deepseek-ai/DeepSeek-R1'
tokenizer = AutoTokenizer.from_pretrained(base_model)


preprocessed_stock_tickers_list = []
for symbol in list(stock_tickers):
  preprocessed_stock_tickers_list.append(preprocess_stock_tickers(symbol))


text = "The stock market saw significant movements today as AAPL and $TSLA reported strong earnings. Meanwhile, BRK.B continued its steady rise, and ABR$D gained investor confidence. However, a surprise drop in GOOG left analysts puzzled. On the other hand, TEST and RANDOM were discussed but are not actual stock tickers."
print(text)

# Without stock tickers tokenizer
tokens = tokenizer.tokenize(text)
print('\nInitial:', tokens)


# With stock tickers tokenizer
preprocessed_text = preprocess_stock_tickers(text)
tokenizer.add_tokens(preprocessed_stock_tickers_list)

stock_indices = [] # i.e. S&P500, NASDAQ. Import from /data/stock-tickers.py
tokenizer.add_tokens(stock_indices)

tokens = tokenizer.tokenize(preprocessed_text)
print('\nWith stock tickers:', tokens)


# Tokenization + Embedding with stock tickers
embeddings = tokenizer(preprocessed_text)
print('\nEmbeddings:', embeddings)


"""
  Not all tokenizers have a space_marker
"""
# Merge tokenizer's space marker and stock tickers
space_marker = tokenizer.tokenize(" a")[0][0]
merged_tokens = merge_space_marker(tokens, space_marker)
print('\nMerged tokens:', merged_tokens)


# Deprocessing (remove stock tickers template <FinGPTICKER_>)
deprocessed_tokens = [reverse_preprocess_stock_ticker(token, space_marker) for token in merged_tokens]
print('\nDeprocessed:', deprocessed_tokens)
