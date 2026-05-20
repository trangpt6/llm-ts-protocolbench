import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)
series = df['Passengers'].astype(float).values

n = len(series)
train_size = int(0.8 * n)
test_size = n - train_size

input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    return X, y

forecasts = []
history = list(series[:train_size])

for t in range(test_size):
    data_arr = np.array(history, dtype=np.float32)
    mn = data_arr.min()
    mx = data_arr.max()
    rng = mx - mn if mx > mn else 1.0
    scaled = (data_arr - mn) / rng

    X_train, y_train = make_sequences(scaled, seq_len)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    # reseed before each retrain for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    n_train = X_train_t.shape[0]
    model.train()
    for ep in range(epochs):
        perm = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_train_t[idx]
            yb = y_train_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()

    last_seq = scaled[-seq_len:].reshape(1, seq_len, 1)
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(last_seq_t).cpu().numpy().flatten()[0]
    pred_val = float(pred_scaled * rng + mn)
    forecasts.append(pred_val)

    history.append(float(series[train_size + t]))

print(forecasts)