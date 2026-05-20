import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use CPU fallback if CUDA is unavailable
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

target_col = "Daily minimum temperatures"
series = df[target_col].astype(float).to_numpy()

train_size = 2920
test_size = 730
train_values = series[:train_size]
test_values = series[train_size:train_size + test_size]

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_dataset(values, seq_len, pred_len):
    X = []
    y = []
    max_i = len(values) - seq_len - pred_len + 1
    for i in range(max_i):
        X.append(values[i:i + seq_len])
        y.append(values[i + seq_len:i + seq_len + pred_len])
    X = np.asarray(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.asarray(y, dtype=np.float32).reshape(-1, pred_len)
    return X, y

def train_model(values):
    X, y = make_dataset(values, seq_len, pred_len)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

history = train_values.astype(float).copy()
forecasts = []

for i in range(test_size):
    model = train_model(history)
    x_input = history[-seq_len:].astype(np.float32).reshape(1, seq_len, 1)
    model.eval()
    with torch.no_grad():
        pred = model(torch.tensor(x_input).to(device)).cpu().numpy().reshape(-1)
    forecasts.append(float(pred[0]))
    history = np.append(history, test_values[i])

print(forecasts)