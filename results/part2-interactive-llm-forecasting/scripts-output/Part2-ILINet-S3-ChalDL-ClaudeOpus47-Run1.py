import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback to CPU when CUDA is unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
series = df['% WEIGHTED ILI'].values.astype(np.float32)

n = len(series)
train_size = int(0.8 * n)
test_size = n - train_size

input_size = 1
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def make_sequences(data, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        Y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    Y = np.array(Y, dtype=np.float32)
    return X, Y

forecasts = []
n_iter = test_size - pred_len + 1
for i in range(n_iter):
    history = series[:train_size + i]
    X_train, Y_train = make_sequences(history, seq_len, pred_len)
    X_train_t = torch.from_numpy(X_train).to(device)
    Y_train_t = torch.from_numpy(Y_train).to(device)
    dataset = TensorDataset(X_train_t, Y_train_t)
    # reset seed each iteration for reproducible retraining
    torch.manual_seed(42)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        last_seq = history[-seq_len:].reshape(1, seq_len, 1).astype(np.float32)
        last_seq_t = torch.from_numpy(last_seq).to(device)
        pred = model(last_seq_t).cpu().numpy().flatten()
    forecasts.extend(pred.tolist())

print(forecasts)