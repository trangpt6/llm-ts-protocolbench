import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv', parse_dates=['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# Features and target
features = df[['Heater', 'Ice cream']].values  # order: Heater (col 0), Ice cream (col 1)
target_col = 1  # Ice cream

# Split
total = len(df)
train_size = int(0.8 * total)  # 158
train_end_idx = train_size - 1  # last train index

# Hyperparameters
seq_len = 6
pred_len = 1
input_size = 2
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # last time step output
        return out

# Function to create sequences from data
def create_sequences(data, seq_len, target_col):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])   # all features
        y.append(data[i+seq_len, target_col])
    return np.array(X), np.array(y)

# Function to train model on given data and return trained model
def train_model(train_data, seq_len, input_size, hidden_size, num_layers, epochs, batch_size, lr):
    X_train, y_train = create_sequences(train_data, seq_len, target_col)
    if len(X_train) == 0:
        # fallback if not enough data: create dummy
        return None
    X_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = GRUModel(input_size, hidden_size, num_layers, output_size=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Initial training data: up to train_end_idx (exclusive, index train_end_idx is last training point)
train_data = features[:train_end_idx+1]  # includes index 0..train_end_idx

# Forecasting: one-step ahead, rolling, retraining at each step
forecasts = []
current_data = train_data.copy()

for step in range(total - train_size):  # 40 steps for test set
    # Train model on current_data
    model = train_model(current_data, seq_len, input_size, hidden_size, num_layers, epochs, batch_size, lr)
    if model is None:
        # Not enough data to train, fallback to naive forecast: last observed value
        forecast = current_data[-1, target_col]
    else:
        model.eval()
        # Use last seq_len steps as input
        last_seq = torch.tensor(current_data[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred = model(last_seq).cpu().item()
        forecast = pred
    
    forecasts.append(forecast)
    
    # Append true observation (if available) to current_data for next step
    # The true observation for this forecasted step is at index train_end_idx+1+step
    true_obs = features[train_end_idx+1+step]  # entire feature vector including Heater and Ice cream
    current_data = np.vstack([current_data, true_obs])

# Print final forecasts as list
print(forecasts)