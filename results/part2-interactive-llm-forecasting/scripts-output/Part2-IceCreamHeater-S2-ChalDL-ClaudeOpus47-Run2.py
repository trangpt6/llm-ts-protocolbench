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

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

total = len(data)
train_size = int(0.8 * total)
test_size = total - train_size

seq_len = 6
pred_len = 1
input_size = 2
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

target_idx = 1

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_sequences(arr, seq_len, target_idx):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len, target_idx])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

forecasts = []

for step in range(test_size):
    available = data[:train_size + step]
    X_train, y_train = make_sequences(available, seq_len, target_idx)

    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1).to(device)

    dataset = TensorDataset(X_train_t, y_train_t)
    g = torch.Generator()
    g.manual_seed(42 + step)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

    # reseed before model init for reproducibility
    torch.manual_seed(42)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
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
    last_seq = data[train_size + step - seq_len:train_size + step]
    x_pred = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        y_pred = model(x_pred).item()
    forecasts.append(float(y_pred))

print(forecasts)