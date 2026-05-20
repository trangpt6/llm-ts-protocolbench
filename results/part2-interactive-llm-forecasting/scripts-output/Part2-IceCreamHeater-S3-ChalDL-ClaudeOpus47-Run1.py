import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'], format='%Y-%m')
df = df.sort_values('Month').reset_index(drop=True)

data = df[['Heater', 'Ice cream']].values.astype(np.float32)
target_idx = 1

n_total = len(data)
train_size = int(0.8 * n_total)
test_size = n_total - train_size

input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_sequences(arr, seq_len, pred_len, target_idx):
    X, Y = [], []
    n = len(arr)
    for i in range(n - seq_len - pred_len + 1):
        X.append(arr[i:i+seq_len])
        Y.append(arr[i+seq_len:i+seq_len+pred_len, target_idx])
    if len(X) == 0:
        return (np.empty((0, seq_len, arr.shape[1]), dtype=np.float32),
                np.empty((0, pred_len), dtype=np.float32))
    return np.array(X, dtype=np.float32), np.array(Y, dtype=np.float32)

forecasts = []

for step in range(test_size):
    train_end = train_size + step
    train_arr = data[:train_end]

    mu = train_arr.mean(axis=0)
    sd = train_arr.std(axis=0)
    sd[sd == 0] = 1.0
    norm_train = (train_arr - mu) / sd

    X, Y = make_sequences(norm_train, seq_len, pred_len, target_idx)

    # reset seeds before each retraining for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    if len(X) > 0:
        X_t = torch.from_numpy(X).to(device)
        Y_t = torch.from_numpy(Y).to(device)
        n_samples = len(X_t)
        model.train()
        for ep in range(epochs):
            perm = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i+batch_size]
                xb = X_t[idx]
                yb = Y_t[idx]
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()

    model.eval()
    last_seq = norm_train[-seq_len:]
    x_in = torch.from_numpy(last_seq[None, :, :]).to(device)
    with torch.no_grad():
        y_pred_norm = model(x_in).cpu().numpy()[0]
    y_pred = y_pred_norm * sd[target_idx] + mu[target_idx]
    forecasts.append(float(y_pred[0]))

print(forecasts)