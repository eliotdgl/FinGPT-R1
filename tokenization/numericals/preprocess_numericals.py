import re

class Numbers_preprocessor:
  def __init__(self):
    """
      Initialize the class with regex patterns for different text formats
    """
    self.number_pattern = re.compile(r"\s([+-]?)(\d{1,}(?:,\d+)*(?:\.\d+)?)(?:\s?((?:K|M|B|T|million|billion|trillion)))?(\.?)\s")

    self.currencies = ["$", "€", "£", "¥"] #"CHF", "CAD", "AUD", "NZD", "CNY", "₹", "₽", "R$", "RZAR", "MXN", "SGD", "HKD", "SEK", "NOK", "₩", "₺"]
    self.regex_currencies = r"(?:\$|\€|\£|\¥)" #|CHF|CAD|AUD|NZD|CNY|\₹|\₽|R\$|RZAR|MXN|SGD|HKD|SEK|NOK|\₩|\₺)"

    # Currency-related patterns
    self.before_currency_pattern = re.compile(r"([+-]?)(" + self.regex_currencies + r")\s?(\d{1,}(?:,\d+)*(?:\.\d+)?)(?:\s?((?:K|M|B|T|million|billion|trillion)))?")
    self.after_currency_pattern = re.compile(r"([+-]?)(\d{1,}(?:,\d+)*(?:\.\d+)?)(?:\s?((?:K|M|B|T|million|billion|trillion)))?\s?(" + self.regex_currencies + ")")

    # Percentage-related patterns
    self.before_percentage_pattern = re.compile(r"([+-]?)((?:percent|%))\s?(\d{1,}(?:,\d+)*(?:\.\d+)?)")
    self.after_percentage_pattern = re.compile(r"([+-]?)(\d{1,}(?:,\d+)*(?:\.\d+)?)\s?((?:percent|%))")

    # Master pattern combining all patterns
    self.master_pattern = re.compile(
            rf"(?P<number>{self.number_pattern.pattern})|"
            rf"(?P<currency_before>{self.before_currency_pattern.pattern})|"
            rf"(?P<currency_after>{self.after_currency_pattern.pattern})|"
            rf"(?P<percent_before>{self.before_percentage_pattern.pattern})|"
            rf"(?P<percent_after>{self.after_percentage_pattern.pattern})"
        )

    self.numericals_dict = {}
    self.number_id = 0


  def _number_format(self, number: str)->str:
    """
      Format a number into a compact form (#, K, M, B, T)
    """
    first_digit = number[0]
    if '.' in number:
      int_part, dec_part = number.split('.', 1)
      if len(int_part) == 1 and first_digit in ['0', '1', '2']:
        int_pattern = first_digit
        dec_parttern = '#' * min(len(dec_part),2)
      elif len(int_part) == 1 and first_digit in ['3', '4']:
        int_pattern = first_digit
        dec_parttern = '#'
      else:
        int_pattern = '#' * len(int_part)
        dec_parttern = None

      if int_part[-3:] == '000' and int_part[-6:] != '000000' and len(int_part[:-3]) in [1,2,3]:
        int_pattern = int_pattern[:-3]
        if dec_parttern is not None:
          return f"{int_pattern}.{dec_parttern}", 'K'
        else:
          return f"{int_pattern}", 'K'
      elif int_part[-6:] == '000000' and int_part[-9:] != '000000' and len(int_part[:-6]) in [1,2,3]:
        int_pattern = int_pattern[:-6]
        if dec_parttern is not None:
          return f"{int_pattern}.{dec_parttern}", 'M'
        else:
          return f"{int_pattern}", 'M'
      elif int_part[-9:] == '000000000' and int_part[-12:] != '000000000' and len(int_part[:-9]) in [1,2,3]:
        int_pattern = int_pattern[:-9]
        if dec_parttern is not None:
          return f"{int_pattern}.{dec_parttern}", 'B'
        else:
          return f"{int_pattern}", 'B'
      elif int_part[-12:] == '000000000000' and len(int_part[:-12]) in [1,2,3]:
        int_pattern = int_pattern[:-12]
        if dec_parttern is not None:
          return f"{int_pattern}.{dec_parttern}", 'T'
        else:
          return f"{int_pattern}", 'T'
      elif dec_parttern is not None:
        return f"{int_pattern}.{dec_parttern}", None
      else:
        return f"{int_pattern}", None

    else:
      if number[-3:] == '000' and number[-6:] != '000000' and len(number[:-3]) in [1,2,3]:
        number = number[:-3]
        return '#' * len(number), 'K'
      elif number[-6:] == '000000' and number[-9:] != '000000' and len(number[:-6]) in [1,2,3]:
        number = number[:-6]
        return '#' * len(number), 'M'
      elif number[-9:] == '000000000' and number[-12:] != '000000000' and len(number[:-9]) in [1,2,3]:
        number = number[:-9]
        return '#' * len(number), 'B'
      elif number[-12:] == '000000000000' and len(number[:-12]) in [1,2,3]:
        number = number[:-12]
        return '#' * len(number), 'T'

      return '#' * len(number), None


  def _number_convert(self, match: re.Match)->str:
    """
      Convert a matched number (with optional sign) into its formatted version
    """
    sign, number, order, dot = match.groups()
    number = number.replace(",", "")
    formatted_number, order_from_zeros = self._number_format(number)

    if sign is None or sign == '+':
      sign = ""
    elif sign == "-":
      sign = "-"

    if order is None:
      if order_from_zeros is not None:
        order = order_from_zeros
      else:
        order = ""

    if dot is None:
      dot = " "

    result = f"{sign}{formatted_number}{order}{dot}"
    self.update_numericals_dict(result, number, sign, order, "")

    return ' ' + result + ' '

  def _before_currency_convert(self, match: re.Match)->str:
    """
      Convert a currency amount with the currency symbol before the number into formatted text
    """
    sign, currency, number, order = match.groups()
    number = number.replace(",", "")
    formatted_number, order_from_zeros = self._number_format(number)
    if sign is None or sign == '+':
      sign = ""
    elif sign == "-":
      sign = "-"

    if order is None:
      if order_from_zeros is not None:
        order = order_from_zeros
      else:
        order = ""
    elif order == "million":
      order = "M"
    elif order == "billion":
      order = "B"
    elif order == "trillion":
      order = "T"

    result = f"{sign}{currency}{formatted_number}{order}"
    self.update_numericals_dict(result, number, sign, order, currency)

    return result

  def _after_currency_convert(self, match: re.Match)->str:
    """
      Convert a currency amount with the currency symbol after the number into formatted text
    """
    sign, number, order, currency = match.groups()
    number = number.replace(",", "")
    formatted_number, order_from_zeros = self._number_format(number)

    if sign is None or sign == '+':
      sign = ""
    elif sign == "-":
      sign = "-"

    if order is None:
      if order_from_zeros is not None:
        order = order_from_zeros
      else:
        order = ""
    elif order == "million":
      order = "M"
    elif order == "billion":
      order = "B"
    elif order == "trillion":
      order = "T"

    result = f"{sign}{currency}{formatted_number}{order}"
    self.update_numericals_dict(result, number, sign, order, currency)

    return result

  def _before_percentage_convert(self, match: re.Match)->str:
    """
      Convert a percentage value with the percentage symbol before the number into formatted text
    """
    sign, percent, number = match.groups()
    number = number.replace(",", "")
    formatted_number, _ = self._number_format(number)

    if sign is None or sign == '+':
      sign = ""
    elif sign == "-":
      sign = "-"

    result = f"{sign}{formatted_number}%"
    self.update_numericals_dict(result, number, sign, "", percent)

    return result

  def _after_percentage_convert(self, match: re.Match)->str:
    """
      Convert a percentage value with the percentage symbol after the number into formatted text
    """
    sign, number, percent = match.groups()
    number = number.replace(",", "")
    formatted_number, _ = self._number_format(number)

    if sign is None or sign == '+':
      sign = ""
    elif sign == "-":
      sign = "-"

    result = f"{sign}{formatted_number}%"
    self.update_numericals_dict(result, number, sign, "", percent)

    return result


  def update_numericals_dict(self, template, value, sign, order, unit):
    self.numericals_dict[self.number_id] = {
        "template": template,
        "value": value,
        "sign": sign,
        "order": order,
        "unit": unit
    }
    self.number_id += 1
    pass

  def preprocess_text(self, text: str)->str:
    """
      Preprocess the input text by applying all formatting rules to numbers, currencies, and percentages
    """
    self.numericals_dict = {}
    self.number_id = 0

    output_text = ""
    last_end = 0

    for match in self.master_pattern.finditer(text):
      start, end = match.span()
      output_text += text[last_end:start]
      last_end = end

      matched_text = match.group()

      if match.lastgroup == 'number':
        submatch = self.number_pattern.match(matched_text)
        output_text += self._number_convert(submatch)
      elif match.lastgroup == 'currency_before':
        submatch = self.before_currency_pattern.match(matched_text)
        output_text += self._before_currency_convert(submatch)
      elif match.lastgroup == 'currency_after':
        submatch = self.after_currency_pattern.match(matched_text)
        output_text += self._after_currency_convert(submatch)
      elif match.lastgroup == 'percent_before':
        submatch = self.before_percentage_pattern.match(matched_text)
        output_text += self._before_percentage_convert(submatch)
      elif match.lastgroup == 'percent_after':
        submatch = self.after_percentage_pattern.match(matched_text)
        output_text += self._after_percentage_convert(submatch)

    output_text += text[last_end:]

    return output_text, self.numericals_dict