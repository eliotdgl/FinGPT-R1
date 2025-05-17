from sentiment_model_class import Sentiment_Analysis_Model
from controlled_environment import TextDataGenerator
from sklearn.model_selection import train_test_split
import pandas as pd
import os
csv_path = r"FinGPT-R1\data\sentiment_analysis_train\FinancialPhraseBank-v1.0\Sentences_AllAgree_processed.csv"

df=pd.read_csv(csv_path)
# Split into train and test sets (e.g., 80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.5, random_state=42)

# Optionally save them to new CSV files
train_df.to_csv('train.csv', index=False)
test_df.to_csv('test.csv', index=False)

df_train = os.path.abspath('train.csv')

#sentiment_model = Sentiment_Analysis_Model()
#dataset = sentiment_model.prepare_dataset(df_train)
#sentiment_model.train(dataset)
#sentiment_model.save()

    # Load and test
#sentiment_model.load()
#sample_text = "Apple gained $120M last quarter."
#predicted_label = sentiment_model.predict(sample_text)
#print(f"Predicted label for: '{sample_text}' → {predicted_label}")

generator=TextDataGenerator()
data=generator.generate_batch()
sentiment_model = Sentiment_Analysis_Model()
dataset = sentiment_model.prepare_dataset(data)
sentiment_model.train(dataset)
sentiment_model.save(base_path="./sentiment_model_test_1", timestamp_name="test_1", keep_last=3)
# Load and test

