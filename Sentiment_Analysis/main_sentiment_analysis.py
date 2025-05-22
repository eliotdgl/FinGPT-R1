import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Sentiment_Analysis.sentiment_model_class import Sentiment_Analysis_Model  # Import the class from new_model.py
from Sentiment_Analysis.data_loader import get_train_test_split


# Initialize the sentiment analysis model
sentiment_model = Sentiment_Analysis_Model(load_model=True)
print("\nModel initialized out.\n")
# Load the saved model from the directory
sentiment_model.load(base_path="Sentiment_Analysis/models/sentiment_model_test_1")
print("\nModel loaded out.\n")
# Test the model with a sample text
sample_text = "Apple lost $120M last quarter."
predicted_label = sentiment_model.predict(sample_text)
print(f"\nPredicted label for: '{sample_text}' → {predicted_label}\n")
