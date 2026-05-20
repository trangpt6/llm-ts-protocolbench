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

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
df = df.set_index('DATE')
target_col = '% WEIGHTED ILI'
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W-SUN')
series = df[target_col].reindex(full_idx).interpolate(method='linear').values.astype(np.float32)

total = len(series)
train_size = int(0.8 * total)
test_size = total - train_size

input_size = 1
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(train_data):
    mean = float(train_data.mean())
    std = float(train_data.std()) + 1e-8
    norm = (train_data - mean) / std
    X, y = make_sequences(norm, seq_len, pred_len)
    X_t = torch.tensor(X).unsqueeze(-1).to(device)
    y_t = torch.tensor(y).to(device)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            opt.step()
    return model, mean, std

forecasts = []
current_train = series[:train_size].copy()
n_blocks = (test_size + pred_len - 1) // pred_len

for b in range(n_blocks):
    model, mean, std = train_model(current_train)
    last_seq = (current_train[-seq_len:] - mean) / std
    x_in = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    model.eval()
    with torch.no_grad():
        pred_norm = model(x_in).cpu().numpy().flatten()
    pred = pred_norm * std + mean
    start = train_size + b * pred_len
    end = min(start + pred_len, train_size + test_size)
    take = end - start
    forecasts.extend([float(v) for v in pred[:take]])
    true_block = series[start:end]
    current_train = np.concatenate([current_train, true_block])

forecasts = forecasts[:test_size]
print(forecasts)