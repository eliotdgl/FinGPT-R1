from sentiment_model_class import Sentiment_Analysis_Model  # Import the class from new_model.py

# Initialize the sentiment analysis model
sentiment_model = Sentiment_Analysis_Model(load_model=False)
print("Model initialized out.")
# Load the saved model from the directory
sentiment_model.load(base_path="./sentiment_model_test_1")
print("Model loaded out.")
# Test the model with a sample text
sample_text = "Apple lost $120M last quarter."
predicted_label = sentiment_model.predict(sample_text)
print(f"Predicted label for: '{sample_text}' → {predicted_label}")
