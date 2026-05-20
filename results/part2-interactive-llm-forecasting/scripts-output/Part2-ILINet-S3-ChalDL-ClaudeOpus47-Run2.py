import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# device fallback to cpu when cuda is unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

all_zero_mask = (df[numeric_cols].fillna(0) == 0).all(axis=1)
df.loc[all_zero_mask, target_col] = np.nan
df[target_col] = df[target_col].interpolate(method='linear', limit_direction='both')
df[target_col] = df[target_col].fillna(method='bfill').fillna(method='ffill')

values = df[target_col].values.astype(np.float32)

train_size = 1040
test_size = 261

input_size = 1
seq_len = 13
pred_len = 4
hidden_size = 32
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

def make_sequences(series, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        Y.append(series[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

forecasts = []
num_iterations = test_size - pred_len + 1

for step in range(num_iterations):
    available = values[:train_size + step]
    X_tr, Y_tr = make_sequences(available, seq_len, pred_len)
    mu = float(available.mean())
    sd = float(available.std()) + 1e-8
    X_n = (X_tr - mu) / sd
    Y_n = (Y_tr - mu) / sd
    X_t = torch.tensor(X_n, dtype=torch.float32).unsqueeze(-1).to(device)
    Y_t = torch.tensor(Y_n, dtype=torch.float32).to(device)

    # reseed per step for deterministic retraining
    torch.manual_seed(42 + step)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    n = X_t.size(0)
    for epoch in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            optimizer.zero_grad()
            out = model(X_t[idx])
            loss = criterion(out, Y_t[idx])
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = available[-seq_len:]
        last_seq_n = (last_seq - mu) / sd
        x_in = torch.tensor(last_seq_n, dtype=torch.float32).view(1, seq_len, 1).to(device)
        pred_n = model(x_in).cpu().numpy().flatten()
        pred = pred_n * sd + mu
    forecasts.extend(pred.tolist())

print(forecasts)