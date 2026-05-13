import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Read data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'])
df = df.sort_values('Month')
ts = df['Passengers'].values.astype(np.float32)

# 80/20 split
n_total = len(ts)
train_size = int(0.8 * n_total)  # 115
test_size = n_total - train_size # 29

# Fixed hyperparameters
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001
input_size = 1  # univariate

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM model definition
class LSTMBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # Take the last timestep output for prediction
        last_out = out[:, -1, :]  # (batch, hidden_size)
        y = self.linear(last_out)
        return y

# Create sliding window datasets
def create_windows(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# Forecast function: use latest seq_len values to predict next pred_len
def forecast_block(model, last_seq, scaler):
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(last_seq.reshape(1, seq_len, input_size), dtype=torch.float32).to(device)
        pred = model(inp).cpu().numpy().flatten()
    # Inverse scaling
    dummy = np.zeros((pred_len, 1))
    dummy[:, 0] = pred
    pred_inv = scaler.inverse_transform(dummy)[:, 0]
    return pred_inv

# Training function
def train_model(train_data, retrain=False):
    # Scale the training data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
    # Create windows
    X, y = create_windows(scaled_data, seq_len, pred_len)
    if len(X) == 0:
        raise ValueError("Not enough data for training")
    X = X.reshape(-1, seq_len, input_size)
    y = y  # shape (n, pred_len)

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMBlock().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            preds = model(batch_X)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
    return model, scaler

# Initial training on first 115 points
initial_train = ts[:train_size]
model, scaler = train_model(initial_train)

# Block-wise rolling forecasting
forecasts = []
current_train = ts[:train_size].copy()  # will expand with ground truth each block
blocks_remaining = test_size  # 29
block_index = 0

while blocks_remaining > 0:
    # The next block horizon (up to pred_len)
    horizon = min(pred_len, blocks_remaining)
    # Get the last seq_len values from current training data (scaled)
    scaler_current = MinMaxScaler()
    scaler_current.fit(current_train.reshape(-1, 1))
    last_seq = scaler_current.transform(current_train[-seq_len:].reshape(-1, 1)).flatten()
    # Forecast entire pred_len steps
    block_pred = forecast_block(model, last_seq, scaler_current)
    # Append only needed horizon
    forecasts.extend(block_pred[:horizon].tolist())
    blocks_remaining -= horizon
    if blocks_remaining <= 0:
        break
    # Obtain ground truth for the just forecasted block (the full 12 steps if horizon==12)
    true_block = ts[train_size + block_index * pred_len : train_size + block_index * pred_len + pred_len]
    # Extend training data
    current_train = np.concatenate([current_train, true_block])
    # Retrain model from scratch using all available data
    model, scaler = train_model(current_train)
    block_index += 1

print(forecasts)