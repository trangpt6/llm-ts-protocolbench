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

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

n = len(data)
train_size = int(0.8 * n)
test_size = n - train_size

input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

target_idx = 1

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(arr, seq_len, pred_len, target_idx):
    X, y = [], []
    end = len(arr) - seq_len - pred_len + 1
    for i in range(end):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len:i+seq_len+pred_len, target_idx])
    if len(X) == 0:
        return np.empty((0, seq_len, arr.shape[1]), dtype=np.float32), np.empty((0, pred_len), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

forecasts = []

for step in range(test_size):
    available = data[:train_size + step]

    mean = available.mean(axis=0)
    std = available.std(axis=0)
    std[std == 0] = 1.0
    norm = (available - mean) / std

    X_train, y_train = make_sequences(norm, seq_len, pred_len, target_idx)

    if len(X_train) == 0:
        forecasts.append(float(available[-1, target_idx]))
        continue

    X_train_t = torch.from_numpy(X_train).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)

    # re-seed before each retraining for deterministic behavior
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = LSTMModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    dataset = TensorDataset(X_train_t, y_train_t)
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = norm[-seq_len:]
        x_inf = torch.from_numpy(last_seq).unsqueeze(0).to(device)
        out = model(x_inf).cpu().numpy().flatten()

    pred_value = out[0] * std[target_idx] + mean[target_idx]
    forecasts.append(float(pred_value))

print(forecasts)