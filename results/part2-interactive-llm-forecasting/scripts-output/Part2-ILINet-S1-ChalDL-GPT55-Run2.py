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

# Use CPU fallback if CUDA is unavailable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
    "input_size": 1,
    "seq_len": 52,
    "pred_len": 1,
    "hidden_size": 64,
    "num_layers": 3,
    "kernel_size": 3,
    "dilations": [1, 2, 4, 8, 16, 32],
    "epochs": 50,
    "batch_size": 32,
    "lr": 0.001
}

df = pd.read_csv(r'../../../data/ILINet.csv')
df["DATE"] = pd.to_datetime(df["DATE"])
df = df.sort_values("DATE").reset_index(drop=True)

target_col = "% WEIGHTED ILI"
values = df[target_col].astype(float).to_numpy()

train_size = 1040
test_size = 261
train_values = values[:train_size]

seq_len = hyperparams["seq_len"]
X_train = []
y_train = []
for i in range(len(train_values) - seq_len):
    X_train.append(train_values[i:i + seq_len])
    y_train.append(train_values[i + seq_len])

X_train = np.array(X_train, dtype=np.float32).reshape(-1, seq_len, 1)
y_train = np.array(y_train, dtype=np.float32).reshape(-1, 1)

train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, batch_size=hyperparams["batch_size"], shuffle=True)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU()
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        in_channels = input_size
        for _ in range(num_layers):
            for dilation in dilations:
                layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation))
                in_channels = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)

model = TCN(
    input_size=hyperparams["input_size"],
    hidden_size=hyperparams["hidden_size"],
    num_layers=hyperparams["num_layers"],
    kernel_size=hyperparams["kernel_size"],
    dilations=hyperparams["dilations"],
    pred_len=hyperparams["pred_len"]
).to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])

model.train()
for _ in range(hyperparams["epochs"]):
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

model.eval()
history = train_values.astype(np.float32).tolist()
forecasts = []
with torch.no_grad():
    for _ in range(test_size):
        x = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
        x_tensor = torch.from_numpy(x).to(device)
        pred = float(model(x_tensor).cpu().numpy().reshape(-1)[0])
        forecasts.append(pred)
        history.append(pred)

print(forecasts)