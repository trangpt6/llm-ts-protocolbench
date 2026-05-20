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

INPUT_SIZE = 1
SEQ_LEN = 52
PRED_LEN = 52
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 30
BATCH_SIZE = 32
LR = 0.001

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

target_col = '% WEIGHTED ILI'
original_dates = [pd.Timestamp(d) for d in df['DATE'].tolist()]
original_set = set(original_dates)

full_idx = pd.date_range(start=df['DATE'].iloc[0], end=df['DATE'].iloc[-1], freq='7D')
df_full = df.set_index('DATE').reindex(full_idx)
df_full[target_col] = df_full[target_col].interpolate(method='linear')
series = df_full[target_col].values.astype(np.float32)
dates_full = list(df_full.index)

train_end_date = original_dates[1039]
train_mask = np.array([d <= train_end_date for d in dates_full])
train_series = series[train_mask]
test_series = series[~train_mask]
test_dates = [d for d, m in zip(dates_full, ~train_mask) if m]

mu = float(train_series.mean())
sigma = float(train_series.std())
if sigma == 0.0:
    sigma = 1.0

class LSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=INPUT_SIZE, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE, PRED_LEN)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_windows(arr, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(arr) - seq_len - pred_len + 1):
        X.append(arr[i:i+seq_len])
        Y.append(arr[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

def train_model(history_norm):
    X, Y = make_windows(history_norm, SEQ_LEN, PRED_LEN)
    X_t = torch.from_numpy(X).unsqueeze(-1).to(device)
    Y_t = torch.from_numpy(Y).to(device)
    model = LSTMModel().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()
    ds = TensorDataset(X_t, Y_t)
    # deterministic shuffling
    g = torch.Generator()
    g.manual_seed(42)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, generator=g)
    model.train()
    for _ in range(EPOCHS):
        for xb, yb in dl:
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
    return model

history = train_series.copy()
all_forecasts = []
n_test = len(test_series)
i = 0
while i < n_test:
    hist_norm = (history - mu) / sigma
    model = train_model(hist_norm)
    last_seq = hist_norm[-SEQ_LEN:].astype(np.float32)
    x = torch.from_numpy(last_seq).view(1, SEQ_LEN, INPUT_SIZE).to(device)
    model.eval()
    with torch.no_grad():
        pred_norm = model(x).cpu().numpy().flatten()
    pred = pred_norm * sigma + mu
    block_end = min(i + PRED_LEN, n_test)
    take = block_end - i
    all_forecasts.extend(pred[:take].tolist())
    history = np.concatenate([history, test_series[i:block_end]])
    i = block_end

forecasts = [float(f) for f, d in zip(all_forecasts, test_dates) if d in original_set]
print(forecasts)