<<<<<<< HEAD
from controlled_environment import TextDataGenerator, TextLabelingEnv, ThresholdedRegressor
import torch

# Step 1: Create data generator and environment
generator = TextDataGenerator()
env = TextLabelingEnv(generator)

# Step 2: Train model (this will also learn thresholds)
model = ThresholdedRegressor(env)  # Pass only env
model.train_regressor_in_env(env)

# Step 3: Predict with learned thresholds
def classify(text):
    vec = env.vectorizer.transform([text]).toarray()
    x = torch.tensor(vec, dtype=torch.float32)
    output = model(x).item()
    if output < model.t1.item():
        return -1
    elif output < model.t2.item():
        return 0
    else:
        return 1

# Example usage
sample_text = "Apple gained $120M last quarter."
predicted_label = classify(sample_text)
print(f"Predicted label for: '{sample_text}' → {predicted_label}")
=======
from controlled_environment import TextDataGenerator, TextLabelingEnv, ThresholdedRegressor
import torch

# Step 1: Create data generator and environment
generator = TextDataGenerator()
env = TextLabelingEnv(generator)

# Step 2: Train model (this will also learn thresholds)
model = ThresholdedRegressor(env)  # Pass only env
model.train_regressor_in_env(env)

# Step 3: Predict with learned thresholds
def classify(text):
    vec = env.vectorizer.transform([text]).toarray()
    x = torch.tensor(vec, dtype=torch.float32)
    output = model(x).item()
    if output < model.t1.item():
        return -1
    elif output < model.t2.item():
        return 0
    else:
        return 1

# Example usage
sample_text = "Apple gained $120M last quarter."
predicted_label = classify(sample_text)
print(f"Predicted label for: '{sample_text}' → {predicted_label}")
>>>>>>> 7f486e595270fbc6a0a4d96143501799d382764a
