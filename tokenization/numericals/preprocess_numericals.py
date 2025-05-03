import re

class Numbers_preprocessor:
  def __init__(self):
    """
      Initialize the class with regex patterns for different text formats
    """
    self.number_pattern = re.compile(r"(\s)([+-]?)(\d{1,}(?:,\d+)*(?:\.\d+)?)(?:\s?(?i:(thousand|thousands|million|millions|billion|billions|trillion|trillions|k|m|b|t|thsnd|mln|mil|mill|bn|bln|bil|bill|t|th)))?(\.|\,)?((?=\s|$))")

    self.currencies = ["$", "€", "£", "¥"] #"CHF", "CAD", "AUD", "NZD", "CNY", "₹", "₽", "R$", "RZAR", "MXN", "SGD", "HKD", "SEK", "NOK", "₩", "₺"]
    self.regex_currencies = r"(?:\$|\€|\£|\¥|(?i:dollars|euros|pounds|yens))" #|CHF|CAD|AUD|NZD|CNY|\₹|\₽|R\$|RZAR|MXN|SGD|HKD|SEK|NOK|\₩|\₺)"

    # Currency-related patterns
    self.before_currency_pattern = re.compile(r"([+-]?)(" + self.regex_currencies + r")\s?(\d{1,}(?:,\d+)*(?:\.\d+)?)(?:\s?(?i:(thousand|thousands|million|millions|billion|billions|trillion|trillions|k|m|b|t|thsnd|mln|mil|mill|bn|bln|bil|bill|t|th)))?")
    self.after_currency_pattern = re.compile(r"([+-]?)(\d{1,}(?:,\d+)*(?:\.\d+)?)(?:\s?(?i:(thousand|thousands|million|millions|billion|billions|trillion|trillions|k|m|b|t|thsnd|mln|mil|mill|bn|bln|bil|bill|t|th)))?\s?(" + self.regex_currencies + ")")

    # Percentage-related patterns
    self.before_percentage_pattern = re.compile(r"([+-]?)((?:percent|%))\s?(\d{1,}(?:,\d+)*(?:\.\d+)?)")
    self.after_percentage_pattern = re.compile(r"([+-]?)(\d{1,}(?:,\d+)*(?:\.\d+)?)\s?((?:percent|%))")

    # Master pattern combining all patterns
    self.master_pattern = re.compile(
            rf"(?P<percent_before>{self.before_percentage_pattern.pattern})|"
            rf"(?P<percent_after>{self.after_percentage_pattern.pattern})|"
            rf"(?P<currency_before>{self.before_currency_pattern.pattern})|"
            rf"(?P<currency_after>{self.after_currency_pattern.pattern})"
    )

    self.numericals_dict = {}
    self.number_id = 0

    self.order_to_int = {
      "": 0,
      "K": 3,
      "M": 6,
      "B": 9,
      "T": 12,
    }

    self.unit_to_int = {
      "": 0,
      "$": 1,
      "£": 2,
      "€": 3,
      "¥": 4,
      "%": 5,
    }


  def _number_format(self, number: str)->str:
    """
      Format a number into a compact form (#, K, M, B, T)
    """
    first_digit = number[0]
    if '.' in number:
      int_part, dec_part = number.split('.', 1)
      print(int_part, dec_part)
      if len(int_part) == 1 and first_digit in ['0', '1', '2']:
        int_pattern = first_digit
        dec_pattern = '#' * min(len(dec_part),2)
      elif len(int_part) == 1 and first_digit in ['3', '4']:
        int_pattern = first_digit
        dec_pattern = '#'
      else:
        int_pattern = '#' * len(int_part)
        dec_pattern = None

      if len(int_pattern) in [4,5,6]:
        int_pattern = int_pattern[:-3]
        if dec_pattern is not None:
          return f"{int_pattern}.{dec_pattern}", 'K'
        else:
          return f"{int_pattern}", 'K'
      elif len(int_pattern) in [7,8,9]:
        int_pattern = int_pattern[:-6]
        if dec_pattern is not None:
          return f"{int_pattern}.{dec_pattern}", 'M'
        else:
          return f"{int_pattern}", 'M'
      elif len(int_pattern) in [10,11,12]:
        int_pattern = int_pattern[:-9]
        if dec_pattern is not None:
          return f"{int_pattern}.{dec_pattern}", 'B'
        else:
          return f"{int_pattern}", 'B'
      elif len(int_pattern) in [13,14,15]:
        int_pattern = int_pattern[:-12]
        if dec_pattern is not None:
          return f"{int_pattern}.{dec_pattern}", 'T'
        else:
          return f"{int_pattern}", 'T'
      elif dec_pattern is not None:
        return f"{int_pattern}.{dec_pattern}", None
      else:
        return f"{int_pattern}", None

    else:
      if len(number) in [4,5,6]:
        number = number[:-3]
        return '#' * len(number), 'K'
      elif len(number) in [7,8,9]:
        number = number[:-6]
        return '#' * len(number), 'M'
      elif len(number) in [10,11,12]:
        number = number[:-9]
        return '#' * len(number), 'B'
      elif len(number) in [13,14,15]:
        number = number[:-12]
        return '#' * len(number), 'T'

      return '#' * len(number), None

  def _check_sign_currency_order(self, sign, currency, order, order_from_zeros):
    # Sign
    if sign is None or sign == '+':
      sign = ""
    elif sign == "-":
      sign = "-"
    
    # Currency
    if currency is not None:
      currency = currency.lower()
      if currency == "dollars":
        currency = "$"
      elif currency == "euros":
        currency = "€"
      elif currency == "pounds":
        currency = "£"
      elif currency == "yens":
        currency = "¥"
    else:
      currency = ""
    
    # Order
    if order is None:
      order = order_from_zeros if order_from_zeros is not None else ""
    else:
      order = order.lower()
      if order in ["thousand", "thousands", "k", "thsnd"]:
        order = "K"
      elif order in ["million", "millions", "m", "mln", "mil", "mill"]:
        order = "M"
      elif order in ["billion", "billions", "b", "bn", "bln", "bil", "bill"]:
        order = "B"
      elif order in ["trillion", "trillions", "t", "tn"]:
        order = "T"
      
      if order_from_zeros is not None:
        if order_from_zeros == "K":
          if order == "K":
            order = "M"
          elif order == "M":
            order = "B"
          elif order == "B":
            order = "T"
        if order_from_zeros == "M":
          if order == "K":
            order = "B"
          elif order == "M":
            order = "T"
        if order_from_zeros == "B":
          if order == "K":
            order = "T"

    return sign, currency, order
  
  def _number_convert(self, match: re.Match)->str:
    """
      Convert a matched number (with optional sign) into its formatted version
    """
    space_before, sign, number, original_order, dot, space_after = match.groups()
    number = number.replace(",", "")
    formatted_number, order_from_zeros = self._number_format(number)

    sign, _, order = self._check_sign_currency_order(sign, None, original_order, order_from_zeros)
    
    if dot is None:
      dot = ""
    if space_before is None:
      space_before = ""
    if space_after is None:
      space_before = ""

    result = f"[FinNUM:{sign}{formatted_number}{order}]{dot}"

    self.update_numericals_dict(result, number, sign, original_order, "")

    return space_before + result + space_after

  def _before_currency_convert(self, match: re.Match)->str:
    """
      Convert a currency amount with the currency symbol before the number into formatted text
    """
    sign, currency, number, order = match.groups()
    number = number.replace(",", "")
    formatted_number, order_from_zeros = self._number_format(number)

    _, _, dict_order = self._check_sign_currency_order(None, None, order, None)
    sign, currency, order = self._check_sign_currency_order(sign, currency, order, order_from_zeros)

    result = f"[FinNUM:{sign}{currency}{formatted_number}{order}]"
    self.update_numericals_dict(result, number, sign, dict_order, currency)

    return result

  def _after_currency_convert(self, match: re.Match)->str:
    """
      Convert a currency amount with the currency symbol after the number into formatted text
    """
    sign, number, order, currency = match.groups()
    number = number.replace(",", "")
    formatted_number, order_from_zeros = self._number_format(number)

    _, _, dict_order = self._check_sign_currency_order(None, None, order, None)
    sign, currency, order = self._check_sign_currency_order(sign, currency, order, order_from_zeros)

    result = f"[FinNUM:{sign}{currency}{formatted_number}{order}]"
    self.update_numericals_dict(result, number, sign, dict_order, currency)

    return result

  def _before_percentage_convert(self, match: re.Match)->str:
    """
      Convert a percentage value with the percentage symbol before the number into formatted text
    """
    sign, percent, number = match.groups()
    number = number.replace(",", "")
    formatted_number, _ = self._number_format(number)
    
    sign, _, _ = self._check_sign_currency_order(sign, None, None, None)

    result = f"[FinNUM:{sign}{formatted_number}%]"
    self.update_numericals_dict(result, number, sign, "", percent)

    return result

  def _after_percentage_convert(self, match: re.Match)->str:
    """
      Convert a percentage value with the percentage symbol after the number into formatted text
    """
    sign, number, percent = match.groups()
    number = number.replace(",", "")
    formatted_number, _ = self._number_format(number)

    sign, _, _ = self._check_sign_currency_order(sign, None, None, None)

    result = f"[FinNUM:{sign}{formatted_number}%]"
    self.update_numericals_dict(result, number, sign, "", percent)

    return result


  def update_numericals_dict(self, template, value, sign, order, unit):
    if sign == "-":
      sign = -1
    else:
      sign = 1
    
    self.numericals_dict[self.number_id] = {
        "value": float(value),
        "sign": sign,
        "order": self.order_to_int.get(order, 0),
        "unit": self.unit_to_int.get(unit, 0)
    }
    self.number_id += 1
    pass

  def preprocess_text(self, text: str)->str:
    """
      Preprocess the input text by applying all formatting rules to numbers, currencies, and percentages
    """
    self.numericals_dict = {}
    self.number_id = 0

    intermediate_text = ""
    last_end = 0

    for match in self.master_pattern.finditer(text):
      start, end = match.span()
      intermediate_text += text[last_end:start]
      last_end = end

      matched_text = match.group()
    
      if match.lastgroup == 'currency_before':
        submatch = self.before_currency_pattern.match(matched_text)
        intermediate_text += self._before_currency_convert(submatch)
      elif match.lastgroup == 'currency_after':
        submatch = self.after_currency_pattern.match(matched_text)
        intermediate_text += self._after_currency_convert(submatch)
      elif match.lastgroup == 'percent_before':
        submatch = self.before_percentage_pattern.match(matched_text)
        intermediate_text += self._before_percentage_convert(submatch)
      elif match.lastgroup == 'percent_after':
        submatch = self.after_percentage_pattern.match(matched_text)
        intermediate_text += self._after_percentage_convert(submatch)

    intermediate_text += text[last_end:]
    
    output_text = self.number_pattern.sub(self._number_convert, intermediate_text)

    return output_text, self.numericals_dict