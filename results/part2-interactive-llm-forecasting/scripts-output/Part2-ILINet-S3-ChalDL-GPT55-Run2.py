import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)
y = df["% WEIGHTED ILI"].astype(float).to_numpy()

train_size = 1040
test_size = 261
train_y = y[:train_size]
test_y = y[train_size:train_size + test_size]

class GRUForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(series, seq_len, pred_len):
    xs = []
    ys = []
    n = len(series)
    for i in range(n - seq_len - pred_len + 1):
        xs.append(series[i:i + seq_len])
        ys.append(series[i + seq_len:i + seq_len + pred_len])
    x_arr = np.asarray(xs, dtype=np.float32).reshape(-1, seq_len, 1)
    y_arr = np.asarray(ys, dtype=np.float32).reshape(-1, pred_len)
    return x_arr, y_arr

def train_model(history):
    torch.manual_seed(42)
    model = GRUForecast(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    x_arr, y_arr = make_sequences(history, seq_len, pred_len)
    x_tensor = torch.tensor(x_arr, dtype=torch.float32)
    y_tensor = torch.tensor(y_arr, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    return model

forecasts = []
history = train_y.copy()

for i in range(test_size):
    model = train_model(history)
    x_input = np.asarray(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x_tensor).cpu().numpy().reshape(-1)
    forecasts.append(float(pred[0]))
    history = np.append(history, test_y[i])

print(forecasts)