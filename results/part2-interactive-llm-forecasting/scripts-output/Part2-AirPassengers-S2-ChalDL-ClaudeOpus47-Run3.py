import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
series = df['Passengers'].astype(float).values

total = len(series)
train_size = int(0.8 * total)
test_size = total - train_size

input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_windows(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, input_size)
    y = np.array(y, dtype=np.float32).reshape(-1, pred_len)
    return X, y

forecasts = []

for step in range(test_size):
    history = series[: train_size + step].astype(np.float32)
    mean = history.mean()
    std = history.std() if history.std() > 0 else 1.0
    norm_hist = (history - mean) / std

    X_train, y_train = make_windows(norm_hist, seq_len, pred_len)
    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)

    torch.manual_seed(0)
    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_samples = X_train_t.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for b in range(0, n_samples, batch_size):
            idx = perm[b:b+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_window = norm_hist[-seq_len:].reshape(1, seq_len, input_size)
        last_window_t = torch.from_numpy(last_window).to(device)
        y_pred_norm = model(last_window_t).cpu().numpy().flatten()[0]
        y_pred = y_pred_norm * std + mean
        forecasts.append(float(y_pred))

print(forecasts)