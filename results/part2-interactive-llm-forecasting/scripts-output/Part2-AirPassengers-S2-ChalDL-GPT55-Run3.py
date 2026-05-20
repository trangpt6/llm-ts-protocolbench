import random
import pandas as pd
import numpy as np
import torch
import torch.nn as nn

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use CUDA if available, otherwise fallback to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df["Passengers"].astype(float).values

train_size = int(0.8 * len(values))
train_values = values[:train_size]
test_values = values[train_size:]

params = {'input_size': 1, 'seq_len': 12, 'pred_len': 1, 'hidden_size': 32, 'num_layers': 1, 'epochs': 10, 'batch_size': 8, 'lr': 0.01}

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

def make_sequences(series, seq_len):
    X, y = [], []
    for i in range(len(series) - seq_len):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y

def train_model(series):
    X, y = make_sequences(series, params["seq_len"])
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

    model = GRUModel(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        pred_len=params["pred_len"]
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    model.train()
    n = X_tensor.shape[0]
    for _ in range(params["epochs"]):
        for start in range(0, n, params["batch_size"]):
            end = start + params["batch_size"]
            xb = X_tensor[start:end]
            yb = y_tensor[start:end]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model

history = train_values.copy().astype(float)
forecasts = []

for i in range(len(test_values)):
    model = train_model(history)
    x_input = np.array(history[-params["seq_len"]:], dtype=np.float32).reshape(1, params["seq_len"], 1)
    x_tensor = torch.tensor(x_input, dtype=torch.float32, device=device)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy().reshape(-1)[0]
    forecasts.append(float(pred))
    history = np.append(history, float(test_values[i]))

print(forecasts)