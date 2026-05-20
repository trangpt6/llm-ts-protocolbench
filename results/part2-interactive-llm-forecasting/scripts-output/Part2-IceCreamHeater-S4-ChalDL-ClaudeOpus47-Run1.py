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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

features = df[['Heater', 'Ice cream']].values.astype(np.float32)
target = df['Ice cream'].values.astype(np.float32)

total = len(df)
train_size = int(0.8 * total)
test_size = total - train_size

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

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_in, n_out, k, dilation):
        super().__init__()
        padding = (k - 1) * dilation
        self.conv1 = nn.Conv1d(n_in, n_out, k, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_out, n_out, k, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_in, n_out, 1) if n_in != n_out else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        dils = (dilations * ((num_layers // len(dilations)) + 1))[:num_layers]
        layers = []
        in_ch = input_size
        for i in range(num_layers):
            layers.append(TemporalBlock(in_ch, hidden_size, kernel_size, dils[i]))
            in_ch = hidden_size
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        return self.fc(y)

def make_windows(feats, tgt, seq_len, pred_len):
    X, Y = [], []
    n = len(feats)
    for i in range(n - seq_len - pred_len + 1):
        X.append(feats[i:i+seq_len])
        Y.append(tgt[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_model(feats_arr, tgt_arr):
    X, Y = make_windows(feats_arr, tgt_arr, seq_len, pred_len)
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
    if len(X) == 0:
        return model
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    ds = TensorDataset(Xt, Yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            opt.zero_grad()
            p = model(xb)
            loss = loss_fn(p, yb)
            loss.backward()
            opt.step()
    return model

forecasts = []
current_end = train_size
while current_end < total:
    feats_train = features[:current_end]
    tgt_train = target[:current_end]
    model = train_model(feats_train, tgt_train)
    model.eval()
    x_input = features[current_end - seq_len:current_end]
    x_input_t = torch.from_numpy(x_input).unsqueeze(0).to(device)
    with torch.no_grad():
        pred = model(x_input_t).cpu().numpy().flatten()
    remaining = total - current_end
    take = min(pred_len, remaining)
    forecasts.extend(pred[:take].tolist())
    current_end += pred_len

forecasts = forecasts[:test_size]
print(forecasts)