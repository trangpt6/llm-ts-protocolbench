import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Use CUDA if available, otherwise fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

values = df["Passengers"].astype(float).to_numpy()
train_size = 115
test_size = 29
train_values = values[:train_size]
test_values = values[train_size:train_size + test_size]

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_dataset(series):
    x_list = []
    y_list = []
    max_start = len(series) - seq_len - pred_len + 1
    for i in range(max_start):
        x_list.append(series[i:i + seq_len])
        y_list.append(series[i + seq_len:i + seq_len + pred_len])
    x = torch.tensor(np.array(x_list), dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(np.array(y_list), dtype=torch.float32)
    return TensorDataset(x, y)

def train_model(series):
    model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len).to(device)
    dataset = make_dataset(series)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model

history = train_values.copy()
forecasts = []
start = 0

while start < test_size:
    model = train_model(history)
    x_input = torch.tensor(history[-seq_len:], dtype=torch.float32).view(1, seq_len, input_size).to(device)
    model.eval()
    with torch.no_grad():
        block_pred = model(x_input).cpu().numpy().reshape(-1)
    remaining = test_size - start
    take = min(pred_len, remaining)
    forecasts.extend(block_pred[:take].tolist())
    history = np.concatenate([history, test_values[start:start + take]])
    start += take

print(forecasts)