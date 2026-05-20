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

df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

input_size = 7
seq_len = 12
pred_len = 24
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

h = df['date'].dt.hour.values
dow = df['date'].dt.dayofweek.values
doy = df['date'].dt.dayofyear.values
ot = df['OT'].values.astype(np.float32)

feat = np.zeros((len(df), 7), dtype=np.float32)
feat[:, 0] = ot
feat[:, 1] = np.sin(2 * np.pi * h / 24).astype(np.float32)
feat[:, 2] = np.cos(2 * np.pi * h / 24).astype(np.float32)
feat[:, 3] = np.sin(2 * np.pi * dow / 7).astype(np.float32)
feat[:, 4] = np.cos(2 * np.pi * dow / 7).astype(np.float32)
feat[:, 5] = np.sin(2 * np.pi * doy / 365).astype(np.float32)
feat[:, 6] = np.cos(2 * np.pi * doy / 365).astype(np.float32)

n = len(df)
train_size = int(0.8 * n)
test_size = n - train_size

train_mean = feat[:train_size].mean(axis=0)
train_std = feat[:train_size].std(axis=0) + 1e-8
feat_norm = (feat - train_mean) / train_std

t_mean = float(ot[:train_size].mean())
t_std = float(ot[:train_size].std() + 1e-8)
ot_norm = (ot - t_mean) / t_std

class LSTMModel(n.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def build_seqs(F, T, sl, pl):
    N = len(F) - sl - pl + 1
    if N <= 0:
        return None, None
    X = np.zeros((N, sl, F.shape[1]), dtype=np.float32)
    Y = np.zeros((N, pl), dtype=np.float32)
    for i in range(N):
        X[i] = F[i:i+sl]
        Y[i] = T[i+sl:i+sl+pl]
    return X, Y

forecasts = []
for step in range(test_size):
    end_idx = train_size + step
    F_avail = feat_norm[:end_idx]
    T_avail = ot_norm[:end_idx]
    X_tr, Y_tr = build_seqs(F_avail, T_avail, seq_len, pred_len)
    # resed for deterministic per-step training
    torch.manual_seed(42)
    model = LSTMModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X_tr).to(device)
    Yt = torch.from_numpy(Y_tr).to(device)
    ds = TensorDataset(Xt, Yt)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        x_in = torch.from_numpy(F_avail[-seq_len:]).unsqueeze(0).to(device)
        p_norm = model(x_in).cpu().numpy().flatten()
    p = p_norm * t_std + t_mean
    forecasts.append(float(p[0]))

print(forecasts)