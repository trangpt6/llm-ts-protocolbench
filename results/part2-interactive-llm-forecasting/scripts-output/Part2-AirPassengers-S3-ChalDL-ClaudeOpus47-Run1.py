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
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
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
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()

class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, dilation):
        super().__init__()
        padding = (ksize - 1) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, ksize, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_ch, out_ch, ksize, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.relu1(self.chomp1(self.conv1(x)))
        out = self.relu2(self.chomp2(self.conv2(out)))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, ksize, dilations, seq_len, pred_len):
        super().__init__()
        d_used = dilations[:num_layers] if len(dilations) >= num_layers else dilations
        layers = []
        in_ch = input_size
        for d in d_used:
            layers.append(TCNBlock(in_ch, hidden_size, ksize, d))
            in_ch = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size * seq_len, pred_len)
    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out.reshape(out.size(0), -1)
        return self.fc(out)

def make_windows(series, sl, pl):
    X, Y = [], []
    for i in range(len(series) - sl - pl + 1):
        X.append(series[i:i+sl])
        Y.append(series[i+sl:i+sl+pl])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

forecasts = []
for k in range(test_size):
    history = values[:train_size + k]
    mu = history.mean()
    sd = history.std() + 1e-8
    norm_hist = (history - mu) / sd
    X, Y = make_windows(norm_hist, seq_len, pred_len)
    # reset seeds for retraining reproducibility
    random.seed(42 + k)
    np.random.seed(42 + k)
    torch.manual_seed(42 + k)
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, seq_len, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    if len(X) > 0:
        X_t = torch.from_numpy(X).unsqueeze(-1).to(device)
        Y_t = torch.from_numpy(Y).to(device)
        ds = TensorDataset(X_t, Y_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
        model.train()
        for ep in range(epochs):
            for xb, yb in dl:
                optimizer.zero_grad()
                out = model(xb)
                loss = loss_fn(out, yb)
                loss.backward()
                optimizer.step()
    model.eval()
    with torch.no_grad():
        last_seq = norm_hist[-seq_len:].astype(np.float32)
        x_in = torch.from_numpy(last_seq).reshape(1, seq_len, 1).to(device)
        pred = model(x_in).cpu().numpy().flatten()
    pred = pred * sd + mu
    forecasts.append(float(pred[0]))

print(forecasts)