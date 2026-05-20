import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

zero_mask = (df['%UNWEIGHTED ILI'] == 0) & (df['ILITOTAL'] == 0) & (df['TOTAL PATIENTS'] == 0)
df.loc[zero_mask, '% WEIGHTED ILI'] = np.nan
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear', limit_direction='both')

target = df['% WEIGHTED ILI'].values.astype(np.float32)

n_total = len(target)
n_train = int(0.8 * n_total)
train_series = target[:n_train].tolist()
test_series = target[n_train:].tolist()

seq_len = 13
pred_len = 1
input_size = 1
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def build_sequences(series):
    arr = np.asarray(series, dtype=np.float32)
    n = len(arr) - seq_len
    X = np.zeros((n, seq_len, 1), dtype=np.float32)
    y = np.zeros((n, 1), dtype=np.float32)
    for i in range(n):
        X[i, :, 0] = arr[i:i+seq_len]
        y[i, 0] = arr[i+seq_len]
    return X, y

def fit_and_predict(history):
    X, y = build_sequences(history)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    # Reset seed for reproducible init each retrain
    torch.manual_seed(42)
    model = GRUNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    n = X_t.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            opt.zero_grad()
            out = model(X_t[idx])
            loss = loss_fn(out, y_t[idx])
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        arr = np.asarray(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
        x = torch.from_numpy(arr).to(device)
        return float(model(x).cpu().numpy().flatten()[0])

forecasts = []
history = list(train_series)
for t in range(len(test_series)):
    yhat = fit_and_predict(history)
    forecasts.append(yhat)
    history.append(test_series[t])

print(forecasts)