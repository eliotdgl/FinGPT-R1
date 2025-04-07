currencies = ["$", "€", "£", "¥", "CHF", "CAD", "AUD", "NZD", "CNY", "₹", "₽", "R$", "RZAR", "MXN", "SGD", "HKD", "SEK", "NOK", "₩", "₺"]

int_hashtags = ['#' * t for t in range(1, 5)]
dec_hashtags = ['#' * t1 + '.' + '#' * t2 for t1 in range(0, 6) for t2 in range(1, 2)]

numbers = [str(i) + hashtags for i in range(1,10) for hashtags in (int_hashtags + dec_hashtags)]
neg_numbers = ['-' + number for number in numbers]
numbers_currencies = [currency + number for currency in currencies for number in numbers]
numbers_percentages = [number + '%' for number in numbers]

numbers_with_orders = [number + order for number in numbers for order in ['', 'K', 'M', 'B', 'T']]
numbers_with_orders_currencies = [currency + number_order for currency in currencies for number_order in numbers_with_orders]

minus_numbers_with_orders = ['-' + number_order for number_order in numbers_with_orders]
minus_numbers_with_orders_currencies = ['-' + number_order_currency for number_order_currency in numbers_with_orders_currencies]
minus_numbers_percentages = ['-' + number for number in numbers_percentages]

numericals_vocab = numbers + neg_numbers + numbers_currencies + numbers_percentages + numbers_with_orders + numbers_with_orders_currencies + minus_numbers_with_orders + minus_numbers_with_orders_currencies + minus_numbers_percentages


# Export numericals
import json
import os

if __name__ == "__main__":
    os.makedirs("tokenization/vocabulary", exist_ok=True)
    with open("tokenization/vocabulary/numericals_vocab.json", "w") as f:
        json.dump(numericals_vocab, f)