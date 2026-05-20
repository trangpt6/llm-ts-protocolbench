import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

n_total = len(data)
n_train = int(0.8 * n_total)
n_test = n_total - n_train

input_size = 2
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 30
batch_size = 16
lr = 0.001
target_idx = 1

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, d):
        super().__init__()
        pad = (k - 1) * d
        self.conv1 = nn.Conv1d(n_in, n_out, k, padding=pad, dilation=d)
        self.chomp1 = Chomp1d(pad)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=pad, dilation=d)
        self.chomp2 = Chomp1d(pad)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu1(self.chomp1(self.conv1(x)))
        out = self.relu2(self.chomp2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, in_size, hid, n_layers, k, dils, p_len):
        super().__init__()
        layers = []
        for i in range(n_layers):
            d = dils[i]
            ch_in = in_size if i == 0 else hid
            layers.append(TemporalBlock(ch_in, hid, k, d))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hid, p_len)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        return self.fc(out)

def make_sequences(arr, s_len, p_len, t_idx):
    X, y = [], []
    for i in range(len(arr) - s_len - p_len + 1):
        X.append(arr[i:i+s_len])
        y.append(arr[i+s_len:i+s_len+p_len, t_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(model, X, y, n_epochs, b_size, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    model.train()
    n = len(X)
    X_t = torch.from_numpy(X).to(device)
    y_t = torch.from_numpy(y).to(device)
    for ep in range(n_epochs):
        perm = torch.randperm(n)
        for i in range(0, n, b_size):
            idx = perm[i:i+b_size]
            xb, yb = X_t[idx], y_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

forecasts = []
current_train = data[:n_train].copy()
block_size = 12
n_blocks = (n_test + block_size - 1) // block_size

for b in range(n_blocks):
    start = b * block_size
    end = min(start + block_size, n_test)
    horizon = end - start

    mean = current_train.mean(axis=0)
    std = current_train.std(axis=0) + 1e-8
    norm_train = (current_train - mean) / std

    X_tr, y_tr = make_sequences(norm_train, seq_len, pred_len, target_idx)

    # reseed per block for reproducibility
    torch.manual_seed(42 + b)
    np.random.seed(42 + b)
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
    train_model(model, X_tr, y_tr, epochs, batch_size, lr)

    model.eval()
    last_seq = norm_train[-seq_len:]
    with torch.no_grad():
        x_in = torch.from_numpy(last_seq).unsqueeze(0).to(device)
        pred_norm = model(x_in).cpu().numpy().flatten()

    pred = pred_norm * std[target_idx] + mean[target_idx]
    forecasts.extend(pred[:horizon].tolist())

    true_block = data[n_train + start:n_train + end]
    current_train = np.vstack([current_train, true_block])

print(forecasts)