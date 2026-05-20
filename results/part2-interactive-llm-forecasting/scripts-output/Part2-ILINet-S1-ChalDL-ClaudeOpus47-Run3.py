import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

raw = pd.read_csv(r'../../../data/ILINet.csv')
raw['DATE'] = pd.to_datetime(raw['DATE'])
raw = raw.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'

full_idx = pd.date_range(start=raw['DATE'].iloc[0], end=raw['DATE'].iloc[-1], freq='7D')
df = raw.set_index('DATE').reindex(full_idx)
df[target_col] = df[target_col].interpolate(method='linear')

train_end = pd.Timestamp('2017-10-01')
test_start = pd.Timestamp('2017-10-08')

train_series = df.loc[df.index <= train_end, target_col].values.astype(np.float32)
test_dates = df.index[df.index >= test_start]
H = len(test_dates)

input_size = 1
seq_len = 52
pred_len = 1
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32][:num_layers]
epochs = 50
batch_size = 32
lr = 0.001

mu = float(train_series.mean())
sd = float(train_series.std() + 1e-8)
train_norm = (train_series - mu) / sd

def build_seq(arr, sl, pl):
    X, y = [], []
    for i in range(len(arr) - sl - pl + 1):
        X.append(arr[i:i+sl])
        y.append(arr[i+sl:i+sl+pl])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X_arr, y_arr = build_seq(train_norm, seq_len, pred_len)
X_t = torch.from_numpy(X_arr).unsqueeze(1)
y_t = torch.from_numpy(y_arr)

class Chomp1d(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
    def forward(self, x):
        return x[:, :, :-self.c].contiguous() if self.c > 0 else x

class TCNBlock(nn.Module):
    def __init__(self, ci, co, k, d):
        super().__init__()
        p = (k - 1) * d
        self.conv1 = nn.Conv1d(ci, co, k, padding=p, dilation=d)
        self.chomp1 = Chomp1d(p)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(co, co, k, padding=p, dilation=d)
        self.chomp2 = Chomp1d(p)
        self.relu2 = nn.ReLU()
        self.down = nn.Conv1d(ci, co, 1) if ci != co else None
        self.relu = nn.ReLU()
    def forward(self, x):
        o = self.relu1(self.chomp1(self.conv1(x)))
        o = self.relu2(self.chomp2(self.conv2(o)))
        res = x if self.down is None else self.down(x)
        return self.relu(o + res)

class TCN(nn.Module):
    def __init__(self):
        super().__init__()
        blocks = []
        ci = input_size
        for i in range(num_layers):
            blocks.append(TCNBlock(ci, hidden_size, kernel_size, dilations[i]))
            ci = hidden_size
        self.net = nn.Sequential(*blocks)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        o = self.net(x)
        o = o[:, :, -1]
        return self.fc(o)

model = TCN().to(device)
opt = torch.optim.Adam(model.parameters(), lr=lr)
crit = nn.MSELoss()

ds = TensorDataset(X_t, y_t)
g = torch.Generator()
g.manual_seed(42)
loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

model.train()
for ep in range(epochs):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        opt.zero_grad()
        pr = model(xb)
        loss = crit(pr, yb)
        loss.backward()
        opt.step()

model.eval()
window = list(train_norm[-seq_len:])
fc_norm = []
with torch.no_grad():
    for _ in range(H):
        x_in = torch.tensor(window[-seq_len:], dtype=torch.float32).view(1, 1, seq_len).to(device)
        pv = model(x_in).cpu().numpy().ravel()[0]
        fc_norm.append(pv)
        window.append(pv)

all_fc = [float(v * sd + mu) for v in fc_norm]

orig_test_dates = set(raw.loc[raw['DATE'] >= test_start, 'DATE'].tolist())
forecasts = [f for f, d in zip(all_fc, test_dates) if d in orig_test_dates]

print(forecasts)