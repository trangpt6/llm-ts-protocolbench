import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)
series = df['% WEIGHTED ILI'].astype(float).values.astype(np.float32)

train_size = 1040
test_size = 261

input_size = 1
seq_len = 13
pred_len = 1
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
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, pred_len)
    return X, y

forecasts = []
for t in range(test_size):
    history = series[:train_size + t]
    X, y = make_sequences(history, seq_len, pred_len)
    X_t = torch.tensor(X).to(device)
    y_t = torch.tensor(y).to(device)

    # per-step reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    g = torch.Generator()
    g.manual_seed(42)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=g)

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
        last_seq = torch.tensor(history[-seq_len:].reshape(1, seq_len, 1), dtype=torch.float32).to(device)
        pred = model(last_seq).cpu().numpy().flatten()[0]
    forecasts.append(float(pred))

print(forecasts)