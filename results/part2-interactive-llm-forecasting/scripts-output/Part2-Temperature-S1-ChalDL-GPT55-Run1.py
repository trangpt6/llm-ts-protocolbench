import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Random seed setting
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Device fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
seq_len = 14
pred_len = 1
hidden_size = 64
num_layers = 3
epochs = 50
batch_size = 32
lr = 0.001
kernel_size = 3
dilations = [1, 2, 4, 8]

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"])
df = df.sort_values("Date").reset_index(drop=True)

target = df["Daily minimum temperatures"].astype(np.float32).to_numpy()

train_size = 2920
test_size = 730
train = target[:train_size]
test = target[train_size:train_size + test_size]

X = []
y = []
for i in range(len(train) - seq_len):
    X.append(train[i:i + seq_len])
    y.append(train[i + seq_len])
X = np.asarray(X, dtype=np.float32).reshape(-1, seq_len, input_size)
y = np.asarray(y, dtype=np.float32).reshape(-1, pred_len)

X_tensor = torch.from_numpy(X)
y_tensor = torch.from_numpy(y)
dataset = TensorDataset(X_tensor, y_tensor)
generator = torch.Generator()
generator.manual_seed(42)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

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
        blocks = []
        in_channels = input_size
        for _ in range(num_layers):
            for dilation in dilations:
                blocks.append(TemporalBlock(in_channels, hidden_size, kernel_size, dilation))
                in_channels = hidden_size
        self.network = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        return self.fc(out)

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

model.train()
for _ in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        preds = model(xb)
        loss = criterion(preds, yb)
        loss.backward()
        optimizer.step()

forecasts = []
history = list(train[-seq_len:].astype(np.float32))

model.eval()
with torch.no_grad():
    for _ in range(test_size):
        x_input = np.asarray(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, input_size)
        x_tensor = torch.from_numpy(x_input).to(device)
        pred = model(x_tensor).cpu().numpy().reshape(-1)[0]
        pred_value = float(pred)
        forecasts.append(pred_value)
        history.append(np.float32(pred_value))

print(forecasts)