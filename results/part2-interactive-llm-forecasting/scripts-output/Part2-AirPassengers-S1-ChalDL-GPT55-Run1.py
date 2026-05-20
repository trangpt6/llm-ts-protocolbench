import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use CUDA if available, otherwise CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df["Month"] = pd.to_datetime(df["Month"])
df = df.sort_values("Month").reset_index(drop=True)

values = df["Passengers"].astype(float).values

train_size = 115
test_size = 29
train_values = values[:train_size]

input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 64
num_layers = 2
epochs = 50
batch_size = 16
lr = 0.001

X_train = []
y_train = []
for i in range(len(train_values) - seq_len):
    X_train.append(train_values[i:i + seq_len])
    y_train.append(train_values[i + seq_len])

X_train = np.array(X_train, dtype=np.float32).reshape(-1, seq_len, input_size)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, pred_len)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for _ in range(epochs):
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

model.eval()
history = list(train_values[-seq_len:])
forecasts = []

with torch.no_grad():
    for _ in range(test_size):
        x = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, input_size)
        x_tensor = torch.tensor(x).to(device)
        pred = model(x_tensor).cpu().numpy().reshape(-1)[0]
        forecasts.append(float(pred))
        history.append(float(pred))

print(forecasts)