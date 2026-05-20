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

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

features = df[['Heater', 'Ice cream']].values.astype(np.float32)
target_idx = 1

n_total = len(df)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

input_size = 2
seq_len = 6
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

train_data = features[:n_train]
mean = train_data.mean(axis=0)
std = train_data.std(axis=0) + 1e-8

def scale(x):
    return (x - mean) / std

def inv_scale_target(y_scaled):
    return y_scaled * std[target_idx] + mean[target_idx]

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def make_sequences(data_scaled, seq_len, pred_len, target_idx):
    X, y = [], []
    for i in range(len(data_scaled) - seq_len - pred_len + 1):
        X.append(data_scaled[i:i+seq_len])
        y.append(data_scaled[i+seq_len:i+seq_len+pred_len, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(model, X, y, epochs, batch_size, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model

forecasts = []

for step in range(n_test):
    available = features[:n_train + step]
    available_scaled = scale(available)
    X_tr, y_tr = make_sequences(available_scaled, seq_len, pred_len, target_idx)

    # reset seeds for reproducibility per retrain
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    model = train_model(model, X_tr, y_tr, epochs, batch_size, lr)

    last_seq = available_scaled[-seq_len:]
    x_in = torch.from_numpy(last_seq[np.newaxis, :, :]).to(device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(x_in).cpu().numpy().flatten()[0]
    pred = float(inv_scale_target(pred_scaled))
    forecasts.append(pred)

print(forecasts)