import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

n = len(data)
train_size = int(0.8 * n)
train_data = data[:train_size]
test_data = data[train_size:]

mean = train_data.mean(axis=0)
std = train_data.std(axis=0)
std = np.where(std == 0, 1.0, std)
train_norm = (train_data - mean) / std

seq_len = 12
pred_len = 1
input_size = 2
hidden_size = 32
num_layers = 2
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 50
batch_size = 16
lr = 0.001

def make_windows(arr, seq_len, pred_len):
    X, y = [], []
    for i in range(len(arr) - seq_len - pred_len + 1):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_train, y_train = make_windows(train_norm, seq_len, pred_len)
X_train_t = torch.from_numpy(X_train).permute(0, 2, 1).float()
y_train_t = torch.from_numpy(y_train).permute(0, 2, 1).float()

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu_out = nn.ReLU()
    def forward(self, x):
        out = self.relu1(self.chomp1(self.conv1(x)))
        out = self.relu2(self.chomp2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu_out(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len, output_size):
        super().__init__()
        layers = []
        in_ch = input_size
        for i in range(num_layers):
            d = dilations[i % len(dilations)]
            layers.append(TCNBlock(in_ch, hidden_size, kernel_size, d))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, output_size * pred_len)
        self.pred_len = pred_len
        self.output_size = output_size
    def forward(self, x):
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out.view(-1, self.output_size, self.pred_len)

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len, input_size).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

model.eval()
window = train_norm[-seq_len:].copy()
forecasts = []
target_idx = 1
target_mean = mean[target_idx]
target_std = std[target_idx]

with torch.no_grad():
    for i in range(len(test_data)):
        x = torch.from_numpy(window).permute(1, 0).unsqueeze(0).float().to(device)
        pred = model(x).cpu().numpy()[0]
        next_step = pred[:, 0]
        ic_pred = float(next_step[target_idx] * target_std + target_mean)
        forecasts.append(ic_pred)
        window = np.vstack([window[1:], next_step.reshape(1, -1)])

print(forecasts)