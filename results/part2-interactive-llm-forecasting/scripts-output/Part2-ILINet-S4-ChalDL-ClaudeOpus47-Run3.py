import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# reproducibility seeds
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

target_col = '% WEIGHTED ILI'
y = df[target_col].astype(float).values.copy()

n_total = len(df)
train_size = int(0.8 * n_total)
test_size = n_total - train_size

y_clean = y.copy()
i = 0
while i < train_size:
    if y_clean[i] == 0.0:
        j = i
        while j < train_size and y_clean[j] == 0.0:
            j += 1
        if j - i >= 4:
            y_clean[i:j] = np.nan
        i = j
    else:
        i += 1

s = pd.Series(y_clean)
s = s.interpolate(method='linear', limit_direction='both')
y_clean = s.values

train_y = y_clean[:train_size]
y_min = float(np.min(train_y))
y_max = float(np.max(train_y))
y_range = y_max - y_min if y_max > y_min else 1.0

def scale(x):
    return (x - y_min) / y_range

def inverse(x):
    return x * y_range + y_min

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

input_size = 1
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

def make_sequences(series, sl, pl):
    X, Y = [], []
    for k in range(len(series) - sl - pl + 1):
        X.append(series[k:k+sl])
        Y.append(series[k+sl:k+sl+pl])
    return np.array(X), np.array(Y)

def train_model(history_scaled):
    X, Y = make_sequences(history_scaled, seq_len, pred_len)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    Y_t = torch.tensor(Y, dtype=torch.float32).to(device)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    ds = TensorDataset(X_t, Y_t)
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

forecasts = []
history = list(y_clean[:train_size])
test_values = list(y_clean[train_size:])

n_blocks = (test_size + pred_len - 1) // pred_len

for b in range(n_blocks):
    hist_arr = np.array(history, dtype=float)
    history_scaled = scale(hist_arr)
    model = train_model(history_scaled)
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(history_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        pred_scaled = model(last_seq).cpu().numpy().flatten()
    pred = inverse(pred_scaled)
    start = b * pred_len
    end = min(start + pred_len, test_size)
    take = end - start
    forecasts.extend([float(v) for v in pred[:take]])
    history.extend(test_values[start:end])

print(forecasts)