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

input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
features = df[['Heater', 'Ice cream']].values.astype(np.float32)
target = df['Ice cream'].values.astype(np.float32)

n = len(df)
train_size = int(0.8 * n)
test_size = n - train_size

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def make_windows(feat_arr, tgt_arr):
    Xs, Ys = [], []
    L = len(feat_arr)
    for i in range(L - seq_len - pred_len + 1):
        Xs.append(feat_arr[i:i+seq_len])
        Ys.append(tgt_arr[i+seq_len:i+seq_len+pred_len])
    return np.array(Xs, dtype=np.float32), np.array(Ys, dtype=np.float32)

forecasts = []
n_origins = test_size - pred_len + 1

for i in range(n_origins):
    end = train_size + i
    X, Y = make_windows(features[:end], target[:end])
    Xt = torch.from_numpy(X).to(device)
    Yt = torch.from_numpy(Y).to(device)
    loader = DataLoader(TensorDataset(Xt, Yt), batch_size=batch_size, shuffle=True)

    # per-iteration seed
    torch.manual_seed(42 + i)
    model = LSTMModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            p = model(xb)
            loss = loss_fn(p, yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        last = torch.from_numpy(features[end-seq_len:end]).unsqueeze(0).to(device)
        pred = model(last).cpu().numpy().flatten()
    forecasts.extend(pred.tolist())

print(forecasts)