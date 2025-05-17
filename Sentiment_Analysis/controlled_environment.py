import random
import numpy as np
import gym
from gym import spaces
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import torch.nn as nn
import torch.optim as optim

class TextDataGenerator:
    def __init__(self, n_classes=3, samples_per_episode=100):
        self.n_classes = n_classes
        self.samples_per_episode = samples_per_episode
        self.templates_one_number = {
            #phrases with neutral sentiment with one number
            0: ["{} keeps the price target at {}.",
                "{} declares quarterly dividend of {} per Share, consistent with previous quarters" ],
            #phrases with negative sentiment with one number
            -1: ["{} lost {} last quarter."],
            #phrases with positive sentiment with one number
            1: ["{} gained {} last quarter."],
        }
        self.templates_two_numbers = {
            #phrases with neutral sentiment with two numbers
            0: [ "{} revenue was {} last quarter and is now {}.",
                 "{} went revenue from {} to {}.","{} growth was {} last quarter and is now {}.",
                 "Increase in total sales {} offset by decline in comparable store sales {}"],
            #phrases with negative sentiment with two numbers
            -1: ["{} revenue went down of {} last quarter",
                "{} capitalization went from {} down to {}.", ],
            #phrases with positive sentiment with two numbers
            1: ["{} revenue went up of {} last quarter", 
                "{} capitalization went from {} up to {}."],
        }
        self.templates_stock_change ={
            
            -1: ["{} {}(-{}%)"],
            1: ["{} {}(+{}%)"]

        }
        self.companies =["The company","Apple", "Google", "Microsoft", "Amazon", "Tesla","IBM", "Intel", "NVIDIA", "AMD", "Qualcomm","Samsung", "Sony", "LG", "Panasonic", "Toshiba"]
        

        self.currencies ={
            "$",
            "£",
            "€",
            "¥"
        }

    def generate_sentence(self):
        company = random.choice(self.companies)
        
        type_of_sentence = random.randint(0, 2)
        currency = random.choice(list(self.currencies))
        
        if type_of_sentence == 0:
            # One number template — neutral if small, positive/negative otherwise
            sentiment=random.choice([-1, 0, 1])
            if sentiment==0:
                number = round(random.uniform(0, 300), 2)
                template = random.choice(self.templates_one_number[0])
                sentence = template.format(company, f"{currency}{abs(number)}M")
                label = 0  # neutral
            elif sentiment==-1:
                number = round(random.uniform(10,200), 2)
                template = random.choice(self.templates_one_number[-1])
                sentence = template.format(company, f"{currency}{abs(number)}M")
                if number < 20:
                    label = 0
                else:
                    label = -1  # negative
            else:
                number = round(random.uniform(10,200), 2)
                template = random.choice(self.templates_one_number[-1])
                sentence = template.format(company, f"{currency}{abs(number)}M")
                if number < 20:
                    label = 0
                else:
                    label = 1 

        elif type_of_sentence == 1:
            # Two number template — compare second to first
            num1 = round(random.uniform(10, 1000), 2)
            num2 = round(random.uniform(10, 1000), 2)
            delta = num2-num1
            if abs(delta) < 10:
                label = 0  # neutra
                template = random.choice(self.templates_two_numbers[0])
                sentence = template.format(company, f"{currency}{num1}", f"{currency}{num2}")
            elif delta > 0:
                label = 1  # positive
                template = random.choice(self.templates_two_numbers[1])
                sentence = template.format(company, f"{currency}{num1}", f"{currency}{num2}")
            else:
                label = -1  # positive
                template = random.choice(self.templates_two_numbers[-1])
                sentence = template.format(company, f"{currency}{num1}", f"{currency}{num2}")

        else:
            # Stock change template — label based on % change
            sentiment = random.choice([-1, 1])
            value= round(random.uniform(0, 100), 2)
            change = round(random.uniform(-10, 10), 2)

            if sentiment == 1:
                template = random.choice(self.templates_stock_change[1])
                sentence = template.format(company, f"{currency}{value}",  f"{abs(change)}")
                if change < 1:
                    label = 0
                else:
                    label = 1
            else:
                template = random.choice(self.templates_stock_change[-1])
                sentence = template.format(company, f"{currency}{value}",  f"{abs(change)}")
                if abs(change) < 1:
                    label = 0
                else:
                    label = -1
        return [sentence, label]


    def generate_batch(self):
        texts = []
        labels = []
        for _ in range(self.samples_per_episode):
            text,label = self.generate_sentence()
            texts.append(text)
            labels.append(label)
        return [texts, labels]
generator=TextDataGenerator()



class TextLabelingEnv(gym.Env):
    def __init__(self, data_generator: TextDataGenerator):
        super(TextLabelingEnv, self).__init__()
        self.data_generator = data_generator
        self.vectorizer = TfidfVectorizer()
        # Pre-fit vectorizer to establish a consistent feature size
        sample_texts, _ = self.data_generator.generate_batch()
        self.vectorizer.fit(sample_texts)

        # Use a dummy transform to set observation space shape
        dummy_vec = self.vectorizer.transform(["sample text"]).toarray()
        self.vector_size = dummy_vec.shape[1]

        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.vector_size,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        self._prepare_data()

    def _prepare_data(self):
        self.texts, self.labels = self.data_generator.generate_batch()
        self.X = self.vectorizer.transform(self.texts).toarray()
        self.current_index = 0

    def reset(self):
        self._prepare_data()
        return self.X[self.current_index]

    def step(self, action):
        true_label = self.labels[self.current_index]
        target_value = float(true_label)
        reward = 1.0 - abs(action[0] - target_value)
        self.current_index += 1
        done = self.current_index >= len(self.X)
        obs = (
            self.X[self.current_index]
            if not done
            else np.zeros_like(self.X[0], dtype=np.float32)
        )
        return obs, reward, done, {}

    def render(self, mode="human"):
        print(f"Text: {self.texts[self.current_index]}, Label: {self.labels[self.current_index]}")

class ThresholdedRegressor(nn.Module):
    def __init__(self, env):
        super().__init__()
        input_dim = env.observation_space.shape[0]
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
        # Thresholds initialized inside range [-1, 1]
        self.t1 = nn.Parameter(torch.tensor(-0.33))
        self.t2 = nn.Parameter(torch.tensor(0.33))

    def forward(self, x):
        y_hat = self.regressor(x)
        return y_hat

    def predict_label(self, y_hat):
        # Convert y_hat to label based on learned thresholds
        with torch.no_grad():
            if y_hat < self.t1:
                return -1
            elif y_hat < self.t2:
                return 0
            else:
                return 1
    def margin_loss(self, y_hat, target, t1, t2, margin=0.1):
    # Encourage output to be in the correct region with some margin
        loss = 0.0

    # Handle scalar y_hat and target
        if y_hat < t1 - margin:
            loss += (t1 - margin - y_hat) ** 2
        elif y_hat > t2 + margin:
            loss += (y_hat - t2 - margin) ** 2
        elif t1 + margin <= y_hat <= t2 - margin:
            loss += (y_hat - target) ** 2

        return loss
    
    
    def train_regressor_in_env(self, env: TextLabelingEnv, episodes=10, lr=1e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)

        for ep in range(episodes):
            obs = env.reset()
            total_loss = 0
            done = False

            while not done:
                obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
                y_hat = self(obs_tensor)  # Use the current instance of ThresholdedRegressor
                action = y_hat.detach().cpu().numpy().flatten()
                next_obs, _, done, _ = env.step(action)

                label = torch.tensor(float(env.labels[env.current_index - 1]))  # Ensure scalar
                loss = self.margin_loss(y_hat.squeeze(), label, self.t1, self.t2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                obs = next_obs
                total_loss += loss.item()

            print(f"Episode {ep + 1}: Loss = {total_loss:.4f}, t1 = {self.t1.item():.2f}, t2 = {self.t2.item():.2f}")
        return self
