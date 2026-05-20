import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df['Passengers'].values.astype(np.float32)

n = len(values)
train_size = int(0.8 * n)
test_size = n - train_size

input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 2
epochs = 10
batch_size = 8
lr = 0.01
kernel_size = 3
dilations = [1, 2, 4, 8]

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, d):
        super().__init__()
        padding = (k - 1) * d
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, padding=padding, dilation=d)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, padding=padding, dilation=d)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        ch_in = input_size
        for i in range(num_layers):
            d = dilations[i % len(dilations)]
            layers.append(TemporalBlock(ch_in, hidden_size, kernel_size, d))
            ch_in = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.fc(out)

def make_sequences(series, seq_len, pred_len):
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32)
    return X, y

forecasts = []

for i in range(test_size):
    history = values[:train_size + i].copy()
    mean = history.mean()
    std = history.std() + 1e-8
    norm_history = (history - mean) / std

    X, y = make_sequences(norm_history, seq_len, pred_len)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    torch.manual_seed(42 + i)
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        last_seq = norm_history[-seq_len:].reshape(1, seq_len, 1).astype(np.float32)
        last_seq_t = torch.from_numpy(last_seq).to(device)
        pred = model(last_seq_t).cpu().numpy().flatten()
        pred = pred * std + mean
    forecasts.append(float(pred[0]))

print(forecasts)