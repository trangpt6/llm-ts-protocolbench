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
torch.set_num_threads(1)

# Use CPU if CUDA is unavailable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    torch.cuda.manual_seed_all(42)

input_size = 2
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 2
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 50
batch_size = 16
lr = 0.001

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]

train_size = 158
test_size = 40

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:train_size + test_size].copy()

train_features = train_df[feature_cols].astype(float).values
feature_mean = train_features.mean(axis=0)
feature_std = train_features.std(axis=0)
feature_std[feature_std == 0.0] = 1.0

target_mean = train_df[target_col].astype(float).values.mean()
target_std = train_df[target_col].astype(float).values.std()
if target_std == 0.0:
    target_std = 1.0

scaled_train_features = (train_features - feature_mean) / feature_std
scaled_train_target = (train_df[target_col].astype(float).values - target_mean) / target_std

X_list = []
y_list = []
for i in range(train_size - seq_len):
    X_list.append(scaled_train_features[i:i + seq_len])
    y_list.append(scaled_train_target[i + seq_len])

X = torch.tensor(np.array(X_list), dtype=torch.float32)
y = torch.tensor(np.array(y_list).reshape(-1, 1), dtype=torch.float32)

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.pad = nn.ConstantPad1d((padding, 0), 0.0)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        out = self.relu(out)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(out + residual)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        in_channels = input_size
        for i in range(num_layers):
            dilation = dilations[i % len(dilations)]
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation))
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
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

history = train_df[feature_cols].astype(float).values.tolist()
forecasts = []

model.eval()
with torch.no_grad():
    for step in range(test_size):
        window = np.array(history[-seq_len:], dtype=float)
        scaled_window = (window - feature_mean) / feature_std
        xb = torch.tensor(scaled_window.reshape(1, seq_len, input_size), dtype=torch.float32).to(device)
        scaled_pred = model(xb).cpu().numpy().reshape(-1)[0]
        pred_value = float(scaled_pred * target_std + target_mean)
        forecasts.append(pred_value)
        heater_value = float(test_df.iloc[step]["Heater"])
        history.append([heater_value, pred_value])

print(forecasts)