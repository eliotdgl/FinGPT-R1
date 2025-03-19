from transformers import AutoTokenizer
import re

stock_tickers = {} # Import stock_tickers from /data/stock-tickers.py

def preprocess_stock_tickers(token: str) -> str:
  """
    Identify stock tickers in text and convert them with special template
  """
  regex_pattern = r'\b[A-Z]+(?:[\.|\$][A-Za-z0-9]+)*\b'
  potential_tickers = re.findall(regex_pattern, token)

  valid_tickers = {ticker for ticker in potential_tickers if ticker in stock_tickers}

  def replace_ticker(match):
    ticker = match.group(0)
    return f"<FinGPTICKER_{ticker}>" if ticker in valid_tickers else ticker

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


text = "The stock market saw significant movements today as AAPL and TSLA reported strong earnings. Meanwhile, BRK.B continued its steady rise, and ABR$D gained investor confidence. However, a surprise drop in GOOG left analysts puzzled. On the other hand, TEST and RANDOM were discussed but are not actual stock tickers."


# Without stock tickers tokenizer
tokens = tokenizer.tokenize(text)
print('Initial:', tokens)


# With stock tickers tokenizer
preprocessed_text = preprocess_stock_tickers(text)
tokenizer.add_tokens(preprocessed_stock_tickers_list)
tokens = tokenizer.tokenize(preprocessed_text)
print('With stock tickers:', tokens)


"""
  Not all tokenizers have a space_marker
"""
# Merge tokenizer's space marker and stock tickers
space_marker = tokenizer.tokenize(" a")[0][0]
merged_tokens = merge_space_marker(tokens, space_marker)
print('Merged tokens:', merged_tokens)


# Deprocessing (remove stock tickers template <FinGPTICKER_>)
deprocessed_tokens = [reverse_preprocess_stock_ticker(token, space_marker) for token in merged_tokens]
print('Deprocessed:', deprocessed_tokens)