import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# set seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# device fallback to cpu if cuda unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
df = df.set_index('Date').asfreq('D')
series = df['Daily minimum temperatures'].astype(float).values

total = len(series)
train_size = int(0.8 * total)
train = series[:train_size].copy()
test = series[train_size:].copy()

input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_windows(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i + seq_len])
        Y.append(data[i + seq_len:i + seq_len + pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    Y = np.array(Y, dtype=np.float32).reshape(-1, pred_len)
    return X, Y

def train_model(data_arr, mean, std):
    norm = (data_arr - mean) / std
    X, Y = make_windows(norm, seq_len, pred_len)
    X_t = torch.from_numpy(X)
    Y_t = torch.from_numpy(Y)
    ds = TensorDataset(X_t, Y_t)
    # deterministic shuffling generator for reproducibility
    g = torch.Generator()
    g.manual_seed(SEED)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    model.train()
    for ep in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model

forecasts = []
current_train = train.copy()
test_len = len(test)
i = 0
while i < test_len:
    mean = float(current_train.mean())
    std = float(current_train.std()) + 1e-8
    # reset seeds before each retrain for reproducibility
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    model = train_model(current_train, mean, std)
    model.eval()
    last_seq = (current_train[-seq_len:] - mean) / std
    x_in = torch.from_numpy(last_seq.astype(np.float32).reshape(1, seq_len, 1)).to(device)
    with torch.no_grad():
        pred = model(x_in).cpu().numpy().flatten()
    pred = pred * std + mean
    block_size = min(pred_len, test_len - i)
    forecasts.extend(pred[:block_size].tolist())
    current_train = np.concatenate([current_train, test[i:i + block_size]])
    i += block_size

print(forecasts)