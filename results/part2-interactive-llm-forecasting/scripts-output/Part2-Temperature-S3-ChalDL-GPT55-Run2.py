import random
import pandas as pd
import numpy as np
import torch
from torch import nn

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use CPU fallback if CUDA is unavailable.
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

target = df["Daily minimum temperatures"].astype(float).to_numpy()
train_size = 2920
test_size = 730

train_values = target[:train_size]
test_values = target[train_size:train_size + test_size]

class GRUForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        last = out[:, -1, :]
        return self.fc(last)

def make_windows(values):
    x_list = []
    y_list = []
    n = len(values)
    last_start = n - seq_len - pred_len
    for i in range(last_start + 1):
        x_list.append(values[i:i + seq_len])
        y_list.append(values[i + seq_len:i + seq_len + pred_len])
    x = torch.tensor(np.array(x_list, dtype=np.float32), device=device).unsqueeze(-1)
    y = torch.tensor(np.array(y_list, dtype=np.float32), device=device)
    return x, y

def train_model(values):
    x_train, y_train = make_windows(np.asarray(values, dtype=np.float32))
    model = GRUForecast(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    n = x_train.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_train[idx]
            yb = y_train[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model

history = train_values.astype(np.float32).tolist()
model = train_model(history)
forecasts = []

for i in range(test_size):
    x_input = torch.tensor(np.array(history[-seq_len:], dtype=np.float32), device=device).reshape(1, seq_len, input_size)
    model.eval()
    with torch.no_grad():
        pred = model(x_input).detach().cpu().numpy().reshape(-1)
    forecasts.append(float(pred[0]))
    history.append(float(test_values[i]))
    if i < test_size - 1:
        model = train_model(history)

print(forecasts)