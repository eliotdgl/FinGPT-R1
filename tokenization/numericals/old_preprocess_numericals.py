import re
from numbers_to_vocab import match_number_to_vocab
from words_to_num import replace_numbers


def is_number(tokens : str) -> bool:
  """
    Check wether the token is a number
  """    
  pattern = r"^[$\€\£\¥]?-?\d+(\.\d+)?([ ]?\d+)?%?$"
  return bool(re.fullmatch(pattern, tokens))

def is_percentage(tokens : str) -> bool:
  return bool('%'in tokens)

def is_currency_amount(tokens : str) -> bool:
   return bool('$' in tokens or '€' in tokens or '£' in tokens or '¥' in tokens)

def round_number(tokens : str, precision =5) -> str:
  """
    Round the number keeping only a given number of digits
    By default, keeps the first half of the digits
    and replaces the rest by #.
    Use it for percentage.
  """
  if precision ==0:
    precision = len(tokens) // 2 +1
  if len(tokens) >=8:
      tokens = tokens[:8]
  if bool(re.search(r"\.", tokens)):
    precision-=1
    [integ, decim] = tokens.split('.', 1) if '.' in tokens else [tokens, ""]
    if len(integ) >= precision:
      result = integ[:precision] + ("#" * (len(integ) - precision))
    elif len(integ) + len(decim) >= precision:
      result = integ + "." + decim[:precision - len(integ)]
    else:
      result = tokens
  else:
    result = tokens[:precision] + ("#" * (len(tokens) - precision))
  return result

def merge_numbers(text):
    tokens = text.split()
    result = []
    buffer = []
    
    for token in tokens:
        if token.isdigit():
            buffer.append(token)
        else:
            if buffer:
                result.append(''.join(buffer))
                buffer = []
            result.append(token)
    
    if buffer:
        result.append(''.join(buffer))
    
    return ' '.join(result)

  

def preproccess_numbers(token :str,splited=False)-> str :
    """
    Identify numbers in text,replace them with their numerical values and round them. 
    Then, if splited, return the preprocessed text, the numerical part and the qualitative part.
    Otherwise, return the preprocessed text.

    """
    temp=str(replace_numbers(token))
    print(temp)
    temp = merge_numbers(temp)
    result=''
    numericals=''
    qualitatives=''
    for word in temp.split():
        if is_number(word):
            if is_percentage(word):
              word2 = round_number(word)
            else:
               word2 = match_number_to_vocab(word)
            numericals+=word2+' '  
            
        else :
          qualitatives+=word+' '
          word2=word
        result+=word2+' '
    if splited:
      return [result, numericals, qualitatives]
    else:
      return result
    
        
"""
print("We can identify numbers:")
print("Is 1234 a number ?", is_number("123"))
print("Is 1.234 a number ?", is_number("1.23"))
print("Is 1 234 a number ?", is_number("1 234"))
print("Is number a number ?", is_number("number"))
print("We round up the length of numbers so our tokenizer can handle them:")
print("Rounded 1234 to", round_number("1234"))
print("Rounded 1.234 to", round_number("1.234"))
print("Rounded 123456.7", round_number("123456.7"))
print("Rounded 1.234567", round_number("1.234567"))
print("We also change numbers written in words to their numeric representation:")
print("test one->",str(replace_numbers("test one")))
print("test twenty two->",str(replace_numbers("test twenty two")))
print("test one hundred and half a thousand->",str(replace_numbers("test one hundred and half a thousand")))
print(preproccess_numbers("test 1234 and 1.234 and 1 234 and 1.234567 and 123456.7"))
"""