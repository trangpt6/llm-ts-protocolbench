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

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
series = df['Daily minimum temperatures'].astype(float).values

total = len(series)
train_size = int(0.8 * total)
test_size = total - train_size

input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def make_windows(data, seq_len, pred_len):
    X, Y = [], []
    n = len(data)
    for i in range(n - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    Y = np.array(Y, dtype=np.float32)
    return X, Y

def train_model(history_arr):
    X, Y = make_windows(history_arr, seq_len, pred_len)
    Xt = torch.from_numpy(X)
    Yt = torch.from_numpy(Y)
    ds = TensorDataset(Xt, Yt)
    # deterministic loader
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
    return model

def predict_block(model, history_arr):
    model.eval()
    x = np.array(history_arr[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
    with torch.no_grad():
        out = model(torch.from_numpy(x).to(device))
    return out.cpu().numpy().flatten().tolist()

forecasts = []
history = list(series[:train_size])
test_data = list(series[train_size:])

idx = 0
while idx < test_size:
    model = train_model(np.array(history, dtype=np.float32))
    block_pred = predict_block(model, history)
    take = min(pred_len, test_size - idx)
    forecasts.extend(block_pred[:take])
    history.extend(test_data[idx:idx+take])
    idx += take

forecasts = [float(v) for v in forecasts]
print(forecasts)