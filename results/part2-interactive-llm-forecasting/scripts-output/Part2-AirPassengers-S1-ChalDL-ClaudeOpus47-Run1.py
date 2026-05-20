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
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

values = df['Passengers'].values.astype(np.float32)
n = len(values)
train_size = int(0.8 * n)
train = values[:train_size]
test = values[train_size:]

input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 64
num_layers = 2
epochs = 50
batch_size = 16
lr = 0.001

train_min = float(train.min())
train_max = float(train.max())

def norm(x):
    return (x - train_min) / (train_max - train_min)

def denorm(x):
    return x * (train_max - train_min) + train_min

train_n = norm(train)

X_list, y_list = [], []
for i in range(len(train_n) - seq_len - pred_len + 1):
    X_list.append(train_n[i:i+seq_len])
    y_list.append(train_n[i+seq_len:i+seq_len+pred_len])
X = np.array(X_list, dtype=np.float32).reshape(-1, seq_len, input_size)
y = np.array(y_list, dtype=np.float32).reshape(-1, pred_len)

X_t = torch.from_numpy(X).to(device)
y_t = torch.from_numpy(y).to(device)

# deterministic DataLoader generator
g = torch.Generator()
g.manual_seed(42)
ds = TensorDataset(X_t, y_t)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

model.train()
for epoch in range(epochs):
    for xb, yb in loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

model.eval()
history = list(train_n[-seq_len:])
forecasts = []
with torch.no_grad():
    for _ in range(len(test)):
        arr = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, input_size)
        x_in = torch.from_numpy(arr).to(device)
        p = float(model(x_in).cpu().numpy().flatten()[0])
        forecasts.append(float(denorm(p)))
        history.append(p)

forecasts = [float(v) for v in forecasts]
print(forecasts)