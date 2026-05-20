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

# Load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE')

# Drop columns with entirely empty data
df = df.drop(columns=['AGE 25-49', 'AGE 25-64'])

# Extract target variable (keep all rows, including zeros)
target = df['% WEIGHTED ILI'].values.astype(np.float32)

# Split exactly as decided: first 1050 train, remaining 263 test
train_size = 1050
train_vals = target[:train_size]
test_vals = target[train_size:]

# Fixed hyperparameters
seq_len = 13
pred_len = 1
input_size = 1
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define GRU model
class GRUNet(nn.Module):
    def __init__(self):
        super(GRUNet, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]  # last hidden state
        out = self.fc(out)
        return out

# Forecast list
forecasts = []

# Rolling forecast with retraining at each step
for i in range(len(test_vals)):
    # Build current training series (train + already observed test points)
    if i == 0:
        current_series = list(train_vals)
    else:
        current_series = list(train_vals) + list(test_vals[:i])

    # Create sliding window sequences
    X, y = [], []
    for j in range(len(current_series) - seq_len - pred_len + 1):
        X.append(current_series[j:j+seq_len])
        y.append(current_series[j+seq_len])  # single step ahead

    # Convert to tensors
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # shape [N, seq_len, 1]
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)  # shape [N, 1]

    # Create DataLoader with shuffle=True
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model, optimizer, loss
    model = GRUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train for fixed epochs
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

    # Forecast next step
    model.eval()
    with torch.no_grad():
        # Use the last seq_len points of current_series as input
        last_seq = torch.tensor(current_series[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)
        pred_val = model(last_seq).item()
    forecasts.append(pred_val)

    # On progress: the true test value becomes available for next retraining
    # (test_vals[i] is already used in current_series for the next iteration automatically)

print(forecasts)