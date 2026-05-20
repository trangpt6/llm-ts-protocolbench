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

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

full_idx = pd.date_range(start=df['DATE'].iloc[0], end=df['DATE'].iloc[-1], freq='7D')
df = df.set_index('DATE').reindex(full_idx)
target = df['% WEIGHTED ILI'].interpolate(method='linear').values.astype(np.float32)

n = len(target)
train_size = int(0.8 * n)
test_size = n - train_size

input_size = 1
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def make_sequences(series, sl, pl):
    X, Y = [], []
    for i in range(len(series) - sl - pl + 1):
        X.append(series[i:i+sl])
        Y.append(series[i+sl:i+sl+pl])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

train_mean = float(target[:train_size].mean())
train_std = float(target[:train_size].std() + 1e-8)

forecasts = []

for i in range(test_size):
    history = target[:train_size + i]
    norm_hist = (history - train_mean) / train_std

    X, Y = make_sequences(norm_hist, seq_len, pred_len)
    X_t = torch.tensor(X).unsqueeze(-1).to(device)
    Y_t = torch.tensor(Y).to(device)

    # reseed each iteration for deterministic retraining
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    n_samples = X_t.shape[0]
    for ep in range(epochs):
        perm = torch.randperm(n_samples)
        for b in range(0, n_samples, batch_size):
            idx = perm[b:b+batch_size]
            xb = X_t[idx]
            yb = Y_t[idx]
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last = norm_hist[-seq_len:]
        x_in = torch.tensor(last, dtype=torch.float32).view(1, seq_len, 1).to(device)
        pred_norm = model(x_in).cpu().numpy().flatten()
    pred_actual = pred_norm * train_std + train_mean
    forecasts.append(float(pred_actual[0]))

print(forecasts)