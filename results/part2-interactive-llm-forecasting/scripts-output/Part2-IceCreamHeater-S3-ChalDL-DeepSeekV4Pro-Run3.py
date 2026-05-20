import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv', parse_dates=['Month'])
df = df.sort_values('Month').reset_index(drop=True)
data_all = df[['Heater', 'Ice cream']].values  # shape (N,2)
n_total = len(df)
train_size = int(0.8 * n_total)  # 158
test_size = n_total - train_size  # 40

# Split index
train_end_idx = train_size - 1  # 157

# Hyperparameters
input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM model definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        _, (h_n, _) = self.lstm(x)
        # h_n: (num_layers, batch, hidden_size)
        out = h_n[-1]  # last layer hidden state
        return self.fc(out)  # (batch, pred_len)

# Forecast list
forecasts = []

# Current known data initially up to train_end_idx
current_end_idx = train_end_idx

# Rolling forecast loop for each test point
for step in range(test_size):
    # Build training sequences from indices 0 to current_end_idx
    data_current = data_all[:current_end_idx+1]  # inclusive
    # Generate samples
    X_list, y_list = [], []
    for i in range(0, current_end_idx+1 - seq_len - pred_len + 1):
        X_list.append(data_current[i:i+seq_len])  # (seq_len, 2)
        # Target: Ice cream column (index 1) for the next pred_len months
        y_list.append(data_current[i+seq_len:i+seq_len+pred_len, 1])
    if len(X_list) == 0:
        # Fallback if not enough data (should not happen with enough training)
        raise ValueError('Not enough training samples')
    X_train = np.array(X_list).astype(np.float32)
    y_train = np.array(y_list).astype(np.float32)

    # Create DataLoader
    dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model and optimizer
    model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Training loop
    model.train()
    for epoch in range(epochs):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()

    # Prepare input for forecasting: last seq_len observations up to current_end_idx
    last_seq = data_all[current_end_idx - seq_len + 1 : current_end_idx + 1]  # shape (seq_len, 2)
    last_seq_tensor = torch.from_numpy(last_seq.astype(np.float32)).unsqueeze(0).to(device)  # (1, seq_len, 2)
    model.eval()
    with torch.no_grad():
        pred_all = model(last_seq_tensor).squeeze(0).cpu().numpy()  # (pred_len,)
    # First step forecast is the prediction for next month
    next_forecast = pred_all[0]
    forecasts.append(next_forecast)

    # Incorporate true test observation (ground truth) into known data
    # Test index = train_size + step
    test_idx = train_size + step
    # Ensure we don't go out of bounds
    if test_idx >= n_total:
        break
    # The true observation (Heater and Ice cream) is already in data_all at test_idx
    # We just extend our current_end_idx to include it
    current_end_idx = test_idx

# Output
print(forecasts)