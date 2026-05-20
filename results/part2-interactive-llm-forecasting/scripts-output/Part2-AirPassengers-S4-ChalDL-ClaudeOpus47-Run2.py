import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df['Passengers'].astype(float).values

n = len(values)
train_size = int(0.8 * n)
test_size = n - train_size
train = values[:train_size].copy()
test = values[train_size:].copy()

input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001
block_size = 12

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_windows(series, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        Y.append(series[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(Y)

def train_model(series):
    mean = series.mean()
    std = series.std() + 1e-8
    s = (series - mean) / std
    X, Y = create_windows(s, seq_len, pred_len)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
    ds = TensorDataset(X_t, Y_t)
    # reproducible shuffling
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model, mean, std

def forecast_block(model, history, mean, std, horizon):
    model.eval()
    s = (np.array(history[-seq_len:]) - mean) / std
    x = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        out = model(x).cpu().numpy().flatten()
    out = out * std + mean
    return out[:horizon].tolist()

forecasts = []
history = list(train)
i = 0
while i < test_size:
    model, mean, std = train_model(np.array(history, dtype=float))
    h = min(block_size, test_size - i)
    block_pred = forecast_block(model, history, mean, std, h)
    forecasts.extend(block_pred)
    history.extend(test[i:i+h].tolist())
    i += h

print(forecasts)