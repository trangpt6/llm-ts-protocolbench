import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

df = pd.read_csv(r'../../../data/ILINet.csv')
df = df[["DATE", "% WEIGHTED ILI"]].copy()
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

y = df["% WEIGHTED ILI"].astype(float).to_numpy()
train_size = 1040
test_size = 261
train_y = y[:train_size]
test_y = y[train_size:train_size + test_size]

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def make_windows(series, seq_len, pred_len):
    xs = []
    ys = []
    n = len(series)
    for i in range(n - seq_len - pred_len + 1):
        xs.append(series[i:i + seq_len])
        ys.append(series[i + seq_len:i + seq_len + pred_len])
    x_arr = np.asarray(xs, dtype=np.float32).reshape(-1, seq_len, 1)
    y_arr = np.asarray(ys, dtype=np.float32)
    return x_arr, y_arr

def train_model(history):
    history = np.asarray(history, dtype=np.float32)
    mean = float(np.mean(history))
    std = float(np.std(history))
    if std == 0.0:
        std = 1.0
    scaled = (history - mean) / std

    x_train, y_train = make_windows(scaled, seq_len, pred_len)
    x_tensor = torch.tensor(x_train, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y_train, dtype=torch.float32, device=device)

    model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    n = x_tensor.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = x_tensor[idx]
            yb = y_tensor[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model, mean, std

forecasts = []
history = train_y.copy()
pos = 0

while pos < test_size:
    torch.manual_seed(42 + pos)
    np.random.seed(42 + pos)
    random.seed(42 + pos)

    model, mean, std = train_model(history)
    last_seq = history[-seq_len:].astype(np.float32)
    last_seq_scaled = (last_seq - mean) / std
    x_input = torch.tensor(last_seq_scaled.reshape(1, seq_len, 1), dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        block_scaled = model(x_input).detach().cpu().numpy().reshape(-1)
    block_forecast = block_scaled * std + mean

    remaining = test_size - pos
    take = min(pred_len, remaining)
    forecasts.extend([float(v) for v in block_forecast[:take]])

    history = np.concatenate([history, test_y[pos:pos + take]])
    pos += take

print(forecasts)