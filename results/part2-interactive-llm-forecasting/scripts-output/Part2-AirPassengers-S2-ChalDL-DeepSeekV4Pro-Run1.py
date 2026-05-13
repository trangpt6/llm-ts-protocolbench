import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
# No preprocessing needed
data = df['Passengers'].values.astype(np.float32)
n = len(data)
train_size = 115
test_size = n - train_size

# Fixed hyperparameters
seq_len = 12
pred_len = 1
input_size = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01

# Define GRU model
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        # Take last time step output
        out = out[:, -1, :]
        out = self.linear(out)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_sequences(arr, seq_len):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len])
    return np.array(X), np.array(y)

def train_model(model, data_array, epochs, batch_size, lr):
    X, y = create_sequences(data_array, seq_len)
    if len(X) == 0:
        return
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        permutation = torch.randperm(len(X_t))
        for i in range(0, len(X_t), batch_size):
            indices = permutation[i:i+batch_size]
            x_batch = X_t[indices]
            y_batch = y_t[indices]
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

# Initialize model and train on training data only
model = GRUModel().to(device)
train_data = data[:train_size]
train_model(model, train_data, epochs, batch_size, lr)

# Prepare for rolling forecast
forecasts = []
current_data = list(data[:train_size].copy())
# Test true values start at index train_size
test_true = data[train_size:train_size+test_size]

for step in range(test_size):
    # Use last seq_len observations for input
    if len(current_data) < seq_len:
        raise ValueError("Not enough history for prediction")
    input_seq = current_data[-seq_len:]
    x = torch.tensor(np.array(input_seq), dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x).item()
    forecasts.append(pred)
    # Add true value to history (ground truth update)
    current_data.append(test_true[step])
    # Retrain from scratch on all available data up to now
    model = GRUModel().to(device)
    train_model(model, np.array(current_data), epochs, batch_size, lr)

print(forecasts)