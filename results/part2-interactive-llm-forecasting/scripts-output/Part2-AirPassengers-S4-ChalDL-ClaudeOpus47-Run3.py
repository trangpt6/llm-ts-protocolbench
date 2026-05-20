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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df['Passengers'].values.astype(np.float32)

total = len(values)
train_size = int(0.8 * total)
test_size = total - train_size

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
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_sequences(series, seq_len, pred_len):
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(series):
    mean = series.mean()
    std = series.std() + 1e-8
    norm = (series - mean) / std
    X, y = make_sequences(norm, seq_len, pred_len)
    X_t = torch.tensor(X).unsqueeze(-1)
    y_t = torch.tensor(y)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model, mean, std

def forecast_block(model, history, mean, std):
    model.eval()
    last = history[-seq_len:]
    norm = (last - mean) / std
    x = torch.tensor(norm, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(x).cpu().numpy().flatten()
    return pred * std + mean

forecasts = []
history = values[:train_size].copy()
remaining = test_size
idx = train_size

while remaining > 0:
    model, mean, std = train_model(history)
    block_pred = forecast_block(model, history, mean, std)
    take = min(block_size, remaining)
    forecasts.extend([float(v) for v in block_pred[:take]])
    history = np.concatenate([history, values[idx:idx+take]])
    idx += take
    remaining -= take

print(forecasts)