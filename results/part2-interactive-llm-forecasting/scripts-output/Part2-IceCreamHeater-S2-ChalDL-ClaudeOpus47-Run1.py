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

df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)

total = len(data)
train_size = int(0.8 * total)
test_size = total - train_size

input_size = 2
seq_len = 6
pred_len = 1
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
        out = self.fc(out)
        return out

def make_sequences(arr, seq_len, target_idx):
    X, y = [], []
    for i in range(len(arr) - seq_len):
        X.append(arr[i:i+seq_len])
        y.append(arr[i+seq_len, target_idx])
    if len(X) == 0:
        return np.empty((0, seq_len, arr.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

forecasts = []

for step in range(test_size):
    end = train_size + step
    available = data[:end]

    X_train, y_train = make_sequences(available, seq_len, target_idx)

    # reseed per retrain for reproducibility
    random.seed(42 + step)
    np.random.seed(42 + step)
    torch.manual_seed(42 + step)

    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    X_t = torch.from_numpy(X_train).to(device)
    y_t = torch.from_numpy(y_train).to(device).unsqueeze(-1)

    n = len(X_t)
    model.train()
    for epoch in range(epochs):
        idx = np.random.permutation(n)
        for i in range(0, n, batch_size):
            b = idx[i:i+batch_size]
            xb = X_t[b]
            yb = y_t[b]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_in = torch.from_numpy(available[-seq_len:]).unsqueeze(0).to(device)
        pred = model(x_in).cpu().numpy().flatten()[0]
    forecasts.append(float(pred))

print(forecasts)