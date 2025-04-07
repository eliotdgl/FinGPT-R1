import re

import json 
    # Load stock indices vocabulary
with open("tokenization/vocabulary/stock_indices_vocab.json", "r") as f:
    tokenized_stock_indices = json.load(f)
    # Load stock tickers vocabulary
with open("tokenization/vocabulary/stock_tickers.json", "r") as f:
    stock_tickers = list(json.load(f))


def preprocess_stock_indices(text: str)->str:
  pattern = r'\$?\b(?:' + '|'.join(re.escape(variant) for variant in tokenized_stock_indices.keys()) + r')\b'
    
  def replace_with_token(match):
    if match.group(0)[0] == '$':
      return tokenized_stock_indices[match.group(0)[1:]]

    return tokenized_stock_indices[match.group(0)]

  preprocessed_text = re.sub(pattern, replace_with_token, text)

  return preprocessed_text


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
        return f"<FinGPTICKER_{ticker}>" if ticker in stock_tickers else '$'+ticker

    return f"<FinGPTICKER_{ticker}>" if ticker in stock_tickers else ticker

  processed_text = re.sub(regex_pattern, replace_ticker, token)

  return processed_text


def preprocess_stocks(text: str)->str:
   preprocessed_stocks_text = preprocess_stock_indices(text)
   preprocessed_stocks_text = preprocess_stock_tickers(preprocessed_stocks_text)
   return preprocessed_stocks_text





"""
def reverse_preprocess_stock_ticker(token: str, space_marker: str = None) -> str:
  #  Convert back special template to stock tickers
  if token.startswith(space_marker + "<FinGPTICKER") and token.endswith(">"):
    return token[0] + token[14:-1]
  return token

  def merge_space_marker(tokens: list[str], space_marker: str = None) -> list[str]:  
  #  Merge tokenized stock tickers and tokenizer's space marker

  merged_tokens = []
  for i, token in enumerate(tokens):
    if token == space_marker and i + 1 < len(tokens) and tokens[i+1].startswith("<FinGPTICKER_"):
      merged_tokens.append(space_marker + tokens[i+1])
    elif not token.startswith("<FinGPTICKER_"):
      merged_tokens.append(token)

  return merged_tokens

#  Not all tokenizers have a space_marker

# Merge tokenizer's space marker and stock tickers
space_marker = tokenizer.tokenize(" a")[0][0]
merged_tokens = merge_space_marker(tokens, space_marker)
print('\nMerged tokens:', merged_tokens)

# Deprocessing (remove stock tickers template <FinGPTICKER_>)
deprocessed_tokens = [reverse_preprocess_stock_ticker(token, space_marker) for token in merged_tokens]
print('\nDeprocessed:', deprocessed_tokens)
"""