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
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
df = df.set_index('Date')
full_idx = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df = df.reindex(full_idx)
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='time')

values = df['Daily minimum temperatures'].values.astype(np.float32)

n_total = len(values)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

input_size = 1
seq_len = 7
pred_len = 1
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
        out = self.fc(out[:, -1, :])
        return out

def make_sequences(arr, seq_len, pred_len):
    X, y = [], []
    for i in range(len(arr) - seq_len - pred_len + 1):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, input_size)
    y = np.array(y, dtype=np.float32).reshape(-1, pred_len)
    return X, y

forecasts = []
history = list(values[:n_train])

for step in range(n_test):
    arr = np.array(history, dtype=np.float32)
    X_train, y_train = make_sequences(arr, seq_len, pred_len)

    # reseed per step for deterministic retraining
    torch.manual_seed(42)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)

    n_samples = len(X_t)
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = torch.from_numpy(arr[-seq_len:].reshape(1, seq_len, input_size)).to(device)
        pred = model(last_seq).cpu().numpy().flatten()[0]

    forecasts.append(float(pred))
    history.append(float(values[n_train + step]))

print(forecasts)