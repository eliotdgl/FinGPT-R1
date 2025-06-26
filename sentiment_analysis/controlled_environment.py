"""
    == controlled_environment.py ==
    Synthetic Financial Sentiment Data Generator
    
    Defines the TextDataGenerator class for generating synthetic financial sentiment-labeled text data.
    Each generated sentence is labeled as positive, negative, or neutral based on predefined templates 
    and on the randomly generated numbers.
"""
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.feature_extraction.text import TfidfVectorizer

class TextDataGenerator:
    """
        A class to generate synthetic financial sentiment-labeled text data.
    """
    def __init__(self, n_classes: int = 3, samples_per_episode: int = 1000, number_of_episode: int = 1):
        self.n_classes = n_classes
        self.samples_per_episode = samples_per_episode
        self.number_of_episode = number_of_episode
        self.templates_one_number = {
            # Sentences with neutral sentiment with one number
            0: ["{} keeps the price target at {}.",
                "{} declares quarterly dividend of {} per Share, consistent with previous quarters" ],
            # Sentences with negative sentiment with one number
            2: ["{} lost {} last quarter."],
            # Sentences with positive sentiment with one number
            1: ["{} gained {} last quarter."],
        }
        self.templates_two_numbers = {
            # Sentences with neutral sentiment with two numbers
            0: [ "{} revenue was {} last quarter and is now {}.",
                 "{} revenue went from {} to {}.","{} growth was {} last quarter and is now {}.",
                 "Increase in total sales {} offset by decline in comparable store sales {}"],
            # Sentences with negative sentiment with two numbers
            2: ["{} revenue went down of {} last quarter",
                "{} capitalization went from {} down to {}.", ],
            # Sentences with positive sentiment with two numbers
            1: ["{} revenue went up of {} last quarter", 
                "{} capitalization went from {} up to {}."],
        }
        self.templates_stock_change = {
            2: ["{} {}(-{}%)"],
            1: ["{} {}(+{}%)"]
        }
        self.companies = ["The company","Apple", "Google", "Microsoft", "Amazon", "Tesla","IBM", "Intel", "NVIDIA", "AMD", "Qualcomm","Samsung", "Sony", "LG", "Panasonic", "Toshiba"]
        self.currencies = {
            "$",
            "£",
            "€",
            "¥"
        }

    def generate_sentence(self) -> list:
        """
            Generates a single synthetic financial sentence and its sentiment label.
        """
        company = random.choice(self.companies)
        
        type_of_sentence = random.randint(0, 2)
        currency = random.choice(list(self.currencies))
        
        if type_of_sentence == 0:
            # One number template — neutral if small, positive/negative otherwise
            sentiment=random.choice([2, 0, 1])
            if sentiment == 0:
                number = round(random.uniform(0, 300), 2)
                template = random.choice(self.templates_one_number[0])
                sentence = template.format(company, f"{currency}{abs(number)}M")
                label = 0  # Neutral
            elif sentiment == 2:
                number = round(random.uniform(10,200), 2)
                template = random.choice(self.templates_one_number[2])
                sentence = template.format(company, f"{currency}{abs(number)}M")
                if number < 20:
                    label = 0 # Neutral
                else:
                    label = 2  # Negative
            else:
                number = round(random.uniform(10,200), 2)
                template = random.choice(self.templates_one_number[2])
                sentence = template.format(company, f"{currency}{abs(number)}M")
                if number < 20:
                    label = 0
                else:
                    label = 1 

        elif type_of_sentence == 1:
            # Two number template — compare second to first
            num1 = round(random.uniform(10, 1000), 2)
            num2 = round(random.uniform(10, 1000), 2)
            delta = num2 - num1
            if abs(delta) < 10:
                label = 0  # Neutral
                template = random.choice(self.templates_two_numbers[0])
                sentence = template.format(company, f"{currency}{num1}", f"{currency}{num2}")
            elif delta > 0:
                label = 1  # Positive
                template = random.choice(self.templates_two_numbers[1])
                sentence = template.format(company, f"{currency}{num1}", f"{currency}{num2}")
            else:
                label = 2  # Negative
                template = random.choice(self.templates_two_numbers[2])
                sentence = template.format(company, f"{currency}{num1}", f"{currency}{num2}")

        else:
            # Stock change template — label based on % change
            sentiment = random.choice([2, 1])
            value = round(random.uniform(50, 300), 2)
            change = round(random.uniform(-5, 5), 2)

            if sentiment == 1:
                template = random.choice(self.templates_stock_change[1])
                sentence = template.format(company, f"{currency}{value}",  f"{abs(change)}")
                if change < 1:
                    label = 0
                else:
                    label = 1
            else:
                template = random.choice(self.templates_stock_change[2])
                sentence = template.format(company, f"{currency}{value}",  f"{abs(change)}")
                if abs(change) < 1:
                    label = 0
                else:
                    label = 2
        return [sentence, label]


    def generate_batch(self) -> list:
        """
            Generates a full batch (episode) of synthetic data.
        """
        texts = []
        labels = []
        for _ in range(self.number_of_episode):
            for _ in range(self.samples_per_episode):
                text,label = self.generate_sentence()
                texts.append(text)
                labels.append(label)

        return [texts, labels]
