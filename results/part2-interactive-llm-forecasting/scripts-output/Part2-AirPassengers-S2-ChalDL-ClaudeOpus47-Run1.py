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
torch.cuda.manual_seed_all(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
series = df['Passengers'].values.astype(np.float32)

total = len(series)
train_size = int(0.8 * total)

input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, pred_len)
    return X, y

forecasts = []

for t in range(train_size, total):
    history = series[:t]
    mean = float(history.mean())
    std = float(history.std())
    if std == 0.0:
        std = 1.0
    norm_history = (history - mean) / std

    X_train, y_train = make_sequences(norm_history, seq_len, pred_len)

    # reset seeds for reproducible retraining at each step
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    dataset = TensorDataset(X_t, y_t)
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = norm_history[-seq_len:].reshape(1, seq_len, 1).astype(np.float32)
        last_seq_t = torch.from_numpy(last_seq).to(device)
        pred_norm = model(last_seq_t).cpu().numpy().flatten()[0]
        pred_value = pred_norm * std + mean
    forecasts.append(float(pred_value))

print(forecasts)