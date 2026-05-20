import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility.
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Use CPU if CUDA is unavailable.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df["Month"] = pd.to_datetime(df["Month"], format="%Y-%m")
df = df.sort_values("Month").reset_index(drop=True)

target_col = "Ice cream"
feature_cols = ["Heater", "Ice cream"]

total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

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

x_scaler = StandardScaler()
y_scaler = StandardScaler()

train_features = train_df[feature_cols].values.astype(np.float32)
train_target = train_df[[target_col]].values.astype(np.float32)

x_scaler.fit(train_features)
y_scaler.fit(train_target)

train_features_scaled = x_scaler.transform(train_features).astype(np.float32)
train_target_scaled = y_scaler.transform(train_target).astype(np.float32)

X_list = []
y_list = []
for i in range(seq_len, len(train_features_scaled) - pred_len + 1):
    X_list.append(train_features_scaled[i - seq_len:i, :])
    y_list.append(train_target_scaled[i:i + pred_len, 0])

X_train = torch.tensor(np.array(X_list), dtype=torch.float32)
y_train = torch.tensor(np.array(y_list), dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_channels = input_size if i == 0 else hidden_size
            dilation = dilations[i % len(dilations)]
            layers.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for _ in range(epochs):
    for xb, yb in train_loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

history_unscaled = train_df[feature_cols].values.astype(np.float32).tolist()
forecasts = []

model.eval()
with torch.no_grad():
    for _ in range(test_size):
        seq_unscaled = np.array(history_unscaled[-seq_len:], dtype=np.float32)
        seq_scaled = x_scaler.transform(seq_unscaled).astype(np.float32)
        x_input = torch.tensor(seq_scaled.reshape(1, seq_len, input_size), dtype=torch.float32).to(device)
        pred_scaled = model(x_input).cpu().numpy().reshape(-1, 1)[0, 0]
        pred_unscaled = y_scaler.inverse_transform(np.array([[pred_scaled]], dtype=np.float32))[0, 0]
        forecasts.append(float(pred_unscaled))
        last_heater = history_unscaled[-1][0]
        history_unscaled.append([float(last_heater), float(pred_unscaled)])

print(forecasts)