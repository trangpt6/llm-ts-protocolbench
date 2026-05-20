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
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
series = df['Daily minimum temperatures'].astype(float).values

n = len(series)
train_size = int(0.8 * n)
test_size = n - train_size

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
        out = out[:, -1, :]
        return self.fc(out)

def make_windows(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

forecasts = []
history = list(series[:train_size])

for step in range(test_size):
    data = np.array(history, dtype=np.float32)
    X_np, y_np = make_windows(data, seq_len, pred_len)
    X = torch.from_numpy(X_np).unsqueeze(-1).to(device)
    y = torch.from_numpy(y_np).to(device)

    # reseed for deterministic retraining
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_samples = X.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb = X[idx]
            yb = y[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = torch.from_numpy(data[-seq_len:]).view(1, seq_len, 1).to(device)
        out = model(last_seq).cpu().numpy().flatten()

    forecasts.append(float(out[0]))
    history.append(float(series[train_size + step]))

print(forecasts)