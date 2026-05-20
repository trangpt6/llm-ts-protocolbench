import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# reproducibility seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
full_idx = pd.date_range(start=df['Date'].iloc[0], end=df['Date'].iloc[-1], freq='D')
df = df.set_index('Date').reindex(full_idx)
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='linear')
series = df['Daily minimum temperatures'].values.astype(np.float32)

total = len(series)
train_size = int(0.8 * total)
test_size = total - train_size

input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def make_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

forecasts = []
for i in range(test_size):
    end = train_size + i
    history = series[:end]
    X, y = make_sequences(history, seq_len, pred_len)
    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)
    y_t = torch.from_numpy(y).to(device)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    n = len(X_t)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n)
        for b in range(0, n, batch_size):
            idx = perm[b:b+batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        last = torch.from_numpy(history[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(last).cpu().numpy().flatten()
    forecasts.append(float(pred[0]))

print(forecasts)