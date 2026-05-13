import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load the raw CSV.
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'], index_col='Month')
series = df['Passengers'].values.astype(np.float32)

# Train/test split.
train_size = 115
test_size = len(series) - train_size  # 29

# Hyperparameters from Turn 2.
seq_len = 12
input_size = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01

# Device.
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define GRU model.
class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        # Use output at last time step.
        last_out = out[:, -1, :]
        return self.linear(last_out)

# Function to create sequence dataset from a 1-D array.
def create_sequences(data):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        seq = data[i:i+seq_len]
        target = data[i+seq_len]
        X.append(seq)
        y.append(target)
    return np.array(X), np.array(y)

# Initialize current training data with the whole train set.
current_train = series[:train_size].copy()
forecasts = []

# Perform rolling one-step ahead forecasting with retraining at each step.
for step in range(test_size):
    # Prepare training sequences.
    X_train, y_train = create_sequences(current_train)
    X_train = torch.tensor(X_train, dtype=torch.float32).reshape(-1, seq_len, 1).to(device)
    y_train = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1).to(device)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = GRUNet().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Train for fixed number of epochs.
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    # Predict next value using the last seq_len observations.
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(current_train[-seq_len:], dtype=torch.float32).reshape(1, seq_len, 1).to(device)
        pred = model(input_seq).item()
    forecasts.append(pred)

    # Append the true test value to expand training data (as per rolling update with ground truth).
    true_value = series[train_size + step]
    current_train = np.append(current_train, true_value).astype(np.float32)

# Print forecasts as a flat Python list.
print(forecasts)