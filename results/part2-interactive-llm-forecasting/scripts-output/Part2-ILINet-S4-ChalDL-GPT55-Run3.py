import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use CUDA if available, otherwise safely fall back to CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

params = {
    "input_size": 1,
    "seq_len": 52,
    "pred_len": 52,
    "hidden_size": 64,
    "num_layers": 2,
    "epochs": 30,
    "batch_size": 32,
    "lr": 0.001,
}

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

target = df["% WEIGHTED ILI"].astype(float).to_numpy()
train_size = 1040
test_size = 261

train_values = target[:train_size].copy()
test_values = target[train_size:train_size + test_size].copy()

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_windows(values, seq_len, pred_len):
    x_list = []
    y_list = []
    n = len(values)
    for i in range(n - seq_len - pred_len + 1):
        x_list.append(values[i:i + seq_len])
        y_list.append(values[i + seq_len:i + seq_len + pred_len])
    x = np.asarray(x_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.asarray(y_list, dtype=np.float32).reshape(-1, pred_len)
    return x, y

def train_and_forecast(history_values):
    seq_len = params["seq_len"]
    pred_len = params["pred_len"]
    x_train, y_train = make_windows(history_values, seq_len, pred_len)

    model = LSTMForecaster(
        input_size=params["input_size"],
        hidden_size=params["hidden_size"],
        num_layers=params["num_layers"],
        pred_len=pred_len,
    ).to(device)

    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=params["batch_size"], shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=params["lr"])
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(params["epochs"]):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    last_x = np.asarray(history_values[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
    with torch.no_grad():
        forecast = model(torch.from_numpy(last_x).to(device)).cpu().numpy().reshape(-1)
    return forecast.astype(float).tolist()

history = train_values.copy()
forecasts = []
start = 0
block_size = 52

while start < test_size:
    block_forecast = train_and_forecast(history)
    remaining = test_size - start
    take = min(block_size, remaining)
    forecasts.extend(block_forecast[:take])
    true_block = test_values[start:start + take]
    history = np.concatenate([history, true_block])
    start += take

forecasts = [float(x) for x in forecasts[:test_size]]
print(forecasts)