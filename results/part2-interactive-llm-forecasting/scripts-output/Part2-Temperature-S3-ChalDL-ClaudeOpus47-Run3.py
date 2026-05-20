import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
values = df['Daily minimum temperatures'].astype(np.float32).values

n = len(values)
train_size = int(0.8 * n)
test_size = n - train_size
train_arr = values[:train_size]
test_arr = values[train_size:]

input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def make_windows(series):
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32)
    return X, y

def train_model(model, X, y):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X).to(device)
    yt = torch.from_numpy(y).to(device)
    m = Xt.size(0)
    for ep in range(epochs):
        idx = torch.randperm(m)
        for s in range(0, m, batch_size):
            b = idx[s:s+batch_size]
            opt.zero_grad()
            pred = model(Xt[b])
            loss = loss_fn(pred, yt[b])
            loss.backward()
            opt.step()

forecasts = []
num_iters = test_size - pred_len + 1
for i in range(num_iters):
    available = np.concatenate([train_arr, test_arr[:i]])
    X, y = make_windows(available)
    model = GRUModel().to(device)
    train_model(model, X, y)
    last_window = available[-seq_len:].astype(np.float32).reshape(1, seq_len, 1)
    with torch.no_grad():
        pred = model(torch.from_numpy(last_window).to(device)).cpu().numpy().flatten()
    forecasts.extend(pred.tolist())

print(forecasts)