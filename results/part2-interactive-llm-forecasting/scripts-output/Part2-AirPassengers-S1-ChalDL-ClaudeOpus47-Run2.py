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

df = pd.read_csv(r'../../../data/AirPassengers.csv')
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

train_mean = float(train.mean())
train_std = float(train.std())
train_norm = (train - train_mean) / train_std

X_list, y_list = [], []
for i in range(len(train_norm) - seq_len):
    X_list.append(train_norm[i:i+seq_len])
    y_list.append(train_norm[i+seq_len:i+seq_len+pred_len])

X_arr = np.array(X_list).reshape(-1, seq_len, input_size)
y_arr = np.array(y_list).reshape(-1, pred_len)

X_t = torch.tensor(X_arr, dtype=torch.float32).to(device)
y_t = torch.tensor(y_arr, dtype=torch.float32).to(device)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
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
for epoch in range(epochs):
    perm = torch.randperm(n_samples)
    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        xb = X_t[idx]
        yb = y_t[idx]
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()

model.eval()
window = list(train_norm[-seq_len:].astype(np.float32))
forecasts = []
with torch.no_grad():
    for _ in range(len(test)):
        x_in = np.array(window[-seq_len:], dtype=np.float32).reshape(1, seq_len, input_size)
        x_in_t = torch.tensor(x_in, dtype=torch.float32).to(device)
        pred_norm = model(x_in_t).cpu().numpy().flatten()[0]
        window.append(float(pred_norm))
        forecasts.append(float(pred_norm * train_std + train_mean))

print(forecasts)