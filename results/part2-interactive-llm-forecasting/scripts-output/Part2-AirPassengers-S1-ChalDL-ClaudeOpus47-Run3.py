import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
series = df['Passengers'].astype(float).values

total = len(series)
train_size = int(0.8 * total)
train = series[:train_size]
test = series[train_size:]

input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 64
num_layers = 2
epochs = 50
batch_size = 16
lr = 0.001

train_min = train.min()
train_max = train.max()
denom = train_max - train_min if train_max > train_min else 1.0
train_scaled = (train - train_min) / denom

X, y = [], []
for i in range(len(train_scaled) - seq_len - pred_len + 1):
    X.append(train_scaled[i:i+seq_len])
    y.append(train_scaled[i+seq_len:i+seq_len+pred_len])
X = np.array(X).reshape(-1, seq_len, input_size)
y = np.array(y).reshape(-1, pred_len)

X_t = torch.tensor(X, dtype=torch.float32).to(device)
y_t = torch.tensor(y, dtype=torch.float32).to(device)

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

n_samples = X_t.shape[0]
model.train()
for ep in range(epochs):
    perm = torch.randperm(n_samples)
    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_t[idx]
        yb = y_t[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

model.eval()
history_scaled = list(train_scaled)
forecasts = []
with torch.no_grad():
    for _ in range(len(test)):
        window = np.array(history_scaled[-seq_len:]).reshape(1, seq_len, input_size)
        wt = torch.tensor(window, dtype=torch.float32).to(device)
        pred_scaled = model(wt).cpu().numpy().flatten()[0]
        history_scaled.append(pred_scaled)
        pred_value = pred_scaled * denom + train_min
        forecasts.append(float(pred_value))

print(forecasts)