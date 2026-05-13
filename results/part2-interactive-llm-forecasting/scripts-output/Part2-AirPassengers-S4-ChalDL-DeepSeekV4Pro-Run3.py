import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv', index_col='Month', parse_dates=True)
series = df['Passengers'].values.astype(np.float32)

# Chronological split: 115 train, 29 test
train_len = 115
test_len = 29
assert len(series) == train_len + test_len

seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)          # out: (batch, seq_len, hidden_size)
        last_hidden = out[:, -1, :]    # (batch, hidden_size)
        return self.fc(last_hidden)

def train_model(model, train_data, epochs, batch_size, lr):
    X, y = create_sequences(train_data, seq_len, pred_len)
    if len(X) == 0:
        return
    X_t = torch.tensor(X.reshape(-1, seq_len, 1), dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        epoch_loss /= len(dataset)

# Block-wise forecasting with retrain
current_train = series[:train_len].copy()
forecasts = []
block_start = train_len
while block_start < len(series):
    remaining = len(series) - block_start
    # Train a new model from scratch on current training data
    model = LSTMForecaster(input_size=1, hidden_size=hidden_size, num_layers=num_layers, pred_len=pred_len)
    train_model(model, current_train, epochs, batch_size, lr)

    # Prepare last sequence
    last_seq = current_train[-seq_len:]
    X_input = torch.tensor(last_seq.reshape(1, seq_len, 1), dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(X_input).numpy().flatten()
    # Use only as many predictions as needed
    take = min(pred_len, remaining)
    block_forecast = pred[:take]
    forecasts.extend(block_forecast.tolist())

    # Ground truth becomes available for this block
    true_block = series[block_start:block_start+take]
    current_train = np.concatenate([current_train, true_block])
    block_start += take

# Ensure the forecast covers exactly the test set
assert len(forecasts) == test_len
print(forecasts)