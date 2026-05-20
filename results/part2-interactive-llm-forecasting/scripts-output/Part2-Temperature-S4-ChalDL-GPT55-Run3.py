import random
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device fallback.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

target = df["Daily minimum temperatures"].astype(float).to_numpy()

train_size = 2920
test_size = 730
train_values = target[:train_size]
test_values = target[train_size:train_size + test_size]

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)

def make_windows(values, seq_len, pred_len):
    x_list = []
    y_list = []
    n = len(values) - seq_len - pred_len + 1
    for i in range(n):
        x_list.append(values[i:i + seq_len])
        y_list.append(values[i + seq_len:i + seq_len + pred_len])
    x = np.asarray(x_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.asarray(y_list, dtype=np.float32).reshape(-1, pred_len)
    return x, y

def train_model(values):
    x, y = make_windows(values, seq_len, pred_len)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
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

forecasts = []
observed = train_values.copy()
start = 0
while start < test_size:
    model = train_model(observed)
    x_input = observed[-seq_len:].astype(np.float32).reshape(1, seq_len, 1)
    x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        block_pred = model(x_tensor).cpu().numpy().reshape(-1)
    block_len = min(pred_len, test_size - start)
    forecasts.extend(block_pred[:block_len].astype(float).tolist())
    true_block = test_values[start:start + block_len]
    observed = np.concatenate([observed, true_block])
    start += block_len

print(forecasts)