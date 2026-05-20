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

df = pd.read_csv(r'../../../data/Temperature.csv')
values = df['Daily minimum temperatures'].astype(float).values

n_total = len(values)
n_train = int(0.8 * n_total)
train = values[:n_train]
test = values[n_train:]

input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

mean = float(train.mean())
std = float(train.std())
if std == 0:
    std = 1.0

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_sequences(series, sl, pl):
    X, Y = [], []
    for i in range(len(series) - sl - pl + 1):
        X.append(series[i:i+sl])
        Y.append(series[i+sl:i+sl+pl])
    return np.array(X), np.array(Y)

def train_model(model, series_norm):
    X, Y = make_sequences(series_norm, seq_len, pred_len)
    if len(X) == 0:
        return model
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
    ds = TensorDataset(X_t, Y_t)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
    return model

history = list(train.astype(float))
model = LSTMModel().to(device)
series_norm = (np.array(history) - mean) / std
train_model(model, series_norm)

forecasts = []
i = 0
test = test.astype(float)
while i < len(test):
    last_seq = np.array(history[-seq_len:])
    last_seq_norm = (last_seq - mean) / std
    x = torch.tensor(last_seq_norm, dtype=torch.float32).view(1, seq_len, 1).to(device)
    model.eval()
    with torch.no_grad():
        out = model(x).cpu().numpy().flatten()
    out_unnorm = out * std + mean
    block_size = min(pred_len, len(test) - i)
    forecasts.extend([float(v) for v in out_unnorm[:block_size]])
    history.extend(test[i:i+block_size].tolist())
    i += block_size
    if i < len(test):
        series_norm = (np.array(history) - mean) / std
        train_model(model, series_norm)

print(forecasts)