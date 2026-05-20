import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

Set random seeds for reproducibility.

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

Use CPU if CUDA is unavailable.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hyperparams = {
    "input_size": 7,
    "seq_len": 48,
    "pred_len": 1,
    "hidden_size": 64,
    "num_layers": 3,
    "kernel_size": 3,
    "dilations": [1, 2, 4, 8, 16],
    "epochs": 30,
    "batch_size": 64,
    "lr": 0.001
}

df = pd.read_csv(r'../../../data/ETTh1.csv')
feature_cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
target_idx = feature_cols.index("OT")

data = df[feature_cols].astype(np.float32).values

train_size = 13936
test_size = 3484
train_data = data[:train_size]
test_data = data[train_size:train_size + test_size]

seq_len = hyperparams["seq_len"]
pred_len = hyperparams["pred_len"]

X_train = []
y_train = []

for i in range(seq_len, train_size - pred_len + 1):
    X_train.append(train_data[i - seq_len:i])
    y_train.append(train_data[i:i + pred_len, target_idx])

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

train_loader = DataLoader(
    TensorDataset(X_train, y_train),
    batch_size=hyperparams["batch_size"],
    shuffle=True
)

class CausalConv1d(nn.Module):
    def init(self, in_channels, out_channels, kernel_size, dilation):
        super().init()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation
        )

def forward(self, x):
    out = self.conv(x)
    if self.padding > 0:
        out = out[:, :, :-self.padding]
    return out

class TCN(nn.Module):
    def init(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().init()
        layers = []
        in_channels = input_size

    for i in range(num_layers):
        dilation = dilations[i]
        layers.append(CausalConv1d(in_channels, hidden_size, kernel_size, dilation))
        layers.append(nn.ReLU())
        in_channels = hidden_size

    self.network = nn.Sequential(*layers)
    self.fc = nn.Linear(hidden_size, pred_len)

def forward(self, x):
    x = x.transpose(1, 2)
    out = self.network(x)
    out = out[:, :, -1]
    out = self.fc(out)
    return out

model = TCN(
    input_size=hyperparams["input_size"],
    hidden_size=hyperparams["hidden_size"],
    num_layers=hyperparams["num_layers"],
    kernel_size=hyperparams["kernel_size"],
    dilations=hyperparams["dilations"],
    pred_len=hyperparams["pred_len"]
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams["lr"])
loss_fn = nn.MSELoss()

model.train()
for _ in range(hyperparams["epochs"]):
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = loss_fn(preds, yb)
        loss.backward()
        optimizer.step()

model.eval()
history = train_data[-seq_len:].copy()
forecasts = []

with torch.no_grad():
    for i in range(test_size):
        x_input = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x_input).cpu().numpy().reshape(-1)[0]
        forecasts.append(float(pred))

    next_row = test_data[i].copy()
    next_row[target_idx] = pred
    history = np.vstack([history, next_row])

print(forecasts)