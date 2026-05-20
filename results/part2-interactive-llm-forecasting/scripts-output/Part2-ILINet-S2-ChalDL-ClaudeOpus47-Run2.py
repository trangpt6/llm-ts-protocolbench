import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

y = df['% WEIGHTED ILI'].values.astype(np.float32)

n_train = 1037
n_test = 260
seq_len = 13
pred_len = 1
input_size = 1
hidden_size = 32
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
        out = self.fc(out[:, -1, :])
        return out

def make_sequences(series, sl, pl):
    X, Y = [], []
    for i in range(len(series) - sl - pl + 1):
        X.append(series[i:i+sl])
        Y.append(series[i+sl:i+sl+pl])
    X = np.array(X, dtype=np.float32).reshape(-1, sl, 1)
    Y = np.array(Y, dtype=np.float32).reshape(-1, pl)
    return X, Y

forecasts = []

for t in range(n_test):
    history = y[:n_train + t]
    X_train, Y_train = make_sequences(history, seq_len, pred_len)
    X_train_t = torch.from_numpy(X_train).to(device)
    Y_train_t = torch.from_numpy(Y_train).to(device)

    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_samples = X_train_t.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples, device=device)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = Y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = history[-seq_len:].reshape(1, seq_len, 1).astype(np.float32)
        last_seq_t = torch.from_numpy(last_seq).to(device)
        pred = model(last_seq_t).cpu().numpy().flatten()[0]
    forecasts.append(float(pred))

print(forecasts)