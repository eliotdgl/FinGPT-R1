currencies = ["$", "€", "£", "¥"] #"CHF", "CAD", "AUD", "NZD", "CNY", "₹", "₽", "R$", "RZAR", "MXN", "SGD", "HKD", "SEK", "NOK", "₩", "₺"]

int_hashtags = ['#' * t for t in range(1, 4)]
dec_hashtags = [str(t1) + '.' + '#' * t2 for t1 in range(0, 3) for t2 in range(1, 3)] + [str(t1) + '.' + '#' for t1 in range(3, 5)]

pos_hashtags = int_hashtags + dec_hashtags
numbers_add_currencies = [currency + number for currency in currencies for number in pos_hashtags]
pos_numbers = pos_hashtags + numbers_add_currencies

numbers_add_percentages = [number + '%' for number in pos_hashtags]
numbers_add_orders = [number + order for number in pos_numbers for order in ['K', 'M', 'B', 'T']]

numbers_add_neg = ['-' + number for number in (numbers_add_percentages+numbers_add_orders+pos_numbers)]

numericals_vocab = pos_numbers + numbers_add_percentages + numbers_add_orders + numbers_add_neg

num_vocab = [f"[FinNUM:{num}]" for num in numericals_vocab]

# Export numericals
import json
import os

if __name__ == "__main__":
    os.makedirs("tokenization/vocabulary", exist_ok=True)
    with open("tokenization/vocabulary/numericals_vocab.json", "w") as f:
        json.dump(num_vocab, f)