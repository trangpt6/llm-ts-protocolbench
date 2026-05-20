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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
series = df['Daily minimum temperatures'].astype(float).values

total = len(series)
train_size = int(0.8 * total)
test_size = total - train_size

input_size = 1
seq_len = 7
pred_len = 1
hidden_size = 16
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
        out = out[:, -1, :]
        return self.fc(out)

def make_sequences(data):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, input_size)
    y = np.array(y, dtype=np.float32).reshape(-1, pred_len)
    return X, y

forecasts = []
history = list(series[:train_size])

for step in range(test_size):
    data = np.array(history, dtype=np.float32)
    X_train, y_train = make_sequences(data)

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device)
    ds = TensorDataset(X_t, y_t)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)

    torch.manual_seed(42 + step)
    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, input_size)
        last_t = torch.from_numpy(last_seq).to(device)
        yhat = model(last_t).cpu().numpy().flatten()[0]
    forecasts.append(float(yhat))

    true_val = float(series[train_size + step])
    history.append(true_val)

print(forecasts)