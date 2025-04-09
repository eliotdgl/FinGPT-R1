import re

def is_number(tokens : str) -> bool:
  """
    Check wether the token is a number
  """    
  return bool(re.fullmatch(r"-?\d+(\.\d+)?([ ]?\d+)?", tokens))

def round_number(tokens : str, precision =0) -> str:
  """
    Round the number keeping only a given number of digits
    By default, keeps the first half of the digits
  """
  if precision ==0:
    precision = len(tokens) // 2 +1
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

def replace_numbers(text):
  """
    Replace numbers in a string with their numeric representation.
  """
  units = {
        'a': 1,
        'zero': 0,
        'one': 1,
        'two': 2,
        'three': 3,
        'four': 4,
        'five': 5,
        'six': 6,
        'seven': 7,
        'eight': 8,
        'nine': 9,
        'ten': 10,
        'eleven': 11,
        'twelve': 12,
        'thirteen': 13,
        'fourteen': 14,
        'fifteen': 15,
        'sixteen': 16,
        'seventeen': 17,
        'eighteen': 18,
        'nineteen': 19
    }
    
  tens = {
        'twenty': 20,
        'thirty': 30,
        'forty': 40,
        'fifty': 50,
        'sixty': 60,
        'seventy': 70,
        'eighty': 80,
        'ninety': 90
    }
    
  hundreds = {'hundred': 100}
    
  scales = {
        'thousand': 1000,
        'million': 1e6,
        'billion': 1e9,
        'trillion': 1e12
    }
    
  fractions = {
        'half': 0.5,
        'quarter': 0.25,
        'third': 1/3,
        'fourth': 0.25,
        'fifth': 0.2,
        'sixth': 1/6,
        'seventh': 1/7,
        'eighth': 0.125,
        'ninth': 1/9,
        'tenth': 0.1
    }
    
  def words_to_num(words):
        # Check for fraction a scale pattern (e.g., 'half a million')
        if len(words) >= 3:
            first, second, third = words[0].lower(), words[1].lower(), words[2].lower()
            if first in fractions and second == 'a' and third in scales:
                return fractions[first] * scales[third]

        total = 0
        current = 0
        for word in words:
            lword = word.lower()
            if lword in units:
                current += units[lword]
            elif lword in tens:
                current += tens[lword]
            elif lword in hundreds:
                if current == 0:
                    current = hundreds[lword]
                else:
                    current *= hundreds[lword]
            elif lword in scales:
                scale = scales[lword]
                if current == 0:
                    current = 1
                total += current * scale
                current = 0
            else:
                return None
        total += current
        return total

  words = text.split()
  i = 0
  new_words = []
  while i < len(words):
        matched = False
        # Check sequences from longest to shortest
        for j in range(len(words), i, -1):
            segment = words[i:j]
            num = words_to_num(segment)
            if num is not None:
                # Format the number appropriately
                if isinstance(num, float):
                    if num.is_integer():
                        new_words.append(str(int(num)))
                    else:
                        new_words.append(str(num))
                else:
                    new_words.append(str(num))
                i = j
                matched = True
                break
        if not matched:
            new_words.append(words[i])
            i += 1
  return ' '.join(new_words)
  

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
            word2 = round_number(word)
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