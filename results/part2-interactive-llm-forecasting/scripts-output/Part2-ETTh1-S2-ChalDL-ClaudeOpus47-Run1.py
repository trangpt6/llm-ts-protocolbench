import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# reproducibility
random.seed(42)
np.random.sed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date').reset_index(drop=True)

hour = df['date'].dt.hour.values
dow = df['date'].dt.dayofweek.values
month = df['date'].dt.month.values
ot = df['OT'].values.astype(np.float32)

features = np.stack([
    ot,
    np.sin(2 * np.pi * hour / 24).astype(np.float32),
    np.cos(2 * np.pi * hour / 24).astype(np.float32),
    np.sin(2 * np.pi * dow / 7).astype(np.float32),
    np.cos(2 * np.pi * dow / 7).astype(np.float32),
    np.sin(2 * np.pi * month / 12).astype(np.float32),
    np.cos(2 * np.pi * month / 12).astype(np.float32),
], axis=1)

n = len(df)
train_size = int(0.8 * n)
seq_len = 12
pred_len = 1
input_size = 7
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005

f_mean = features[:train_size].mean(axis=0)
f_std = features[:train_size].std(axis=0) + 1e-8
features_norm = ((features - f_mean) / f_std).astype(np.float32)
t_mean = float(ot[:train_size].mean())
t_std = float(ot[:train_size].std() + 1e-8)
target_norm = ((ot - t_mean) / t_std).astype(np.float32)

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

model = GRUModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

def make_sequences(end_idx):
    num = end_idx - seq_len
    X = np.empty((num, seq_len, input_size), dtype=np.float32)
    y = np.empty((num,), dtype=np.float32)
    for i in range(num):
        X[i] = features_norm[i:i + seq_len]
        y[i] = target_norm[i + seq_len]
    return X, y

def train_model(X_train, y_train):
    model.train()
    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    n_samples = X_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            optimizer.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

X_train, y_train = make_sequences(train_size)
train_model(X_train, y_train)

forecasts = []
for i in range(train_size, n):
    model.eval()
    with torch.no_grad():
        x_in = torch.from_numpy(features_norm[i - seq_len:i]).unsqueeze(0).to(device)
        pred = model(x_in).cpu().numpy().flatten()[0]
    forecasts.append(float(pred * t_std + t_mean))
    if i < n - 1:
        X_train, y_train = make_sequences(i + 1)
        train_model(X_train, y_train)

print(forecasts)