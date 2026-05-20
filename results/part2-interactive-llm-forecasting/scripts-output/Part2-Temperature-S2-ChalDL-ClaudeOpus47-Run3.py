import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date').reset_index(drop=True)
df = df.set_index('Date')
full_idx = pd.date_range(df.index.min(), df.index.max(), freq='D')
df = df.reindex(full_idx)
df['Daily minimum temperatures'] = df['Daily minimum temperatures'].interpolate(method='linear')

values = df['Daily minimum temperatures'].values.astype(np.float32)

total = len(values)
train_size = int(0.8 * total)
train_data = values[:train_size]
test_data = values[train_size:]

input_size = 1
seq_len = 7
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        return self.fc(out)

def make_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+1])
    X = np.array(X, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.array(y, dtype=np.float32).reshape(-1, 1)
    return X, y

def train_model(model, X, y, epochs, batch_size, lr):
    model.train()
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    ds = TensorDataset(X_t, y_t)
    # Reproducible shuffling
    g = torch.Generator()
    g.manual_seed(42)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for _ in range(epochs):
        for xb, yb in dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model

model = GRUModel().to(device)

X_train, y_train = make_sequences(train_data, seq_len)
model = train_model(model, X_train, y_train, epochs, batch_size, lr)

history = list(train_data.astype(np.float32))
forecasts = []
for t in range(len(test_data)):
    model.eval()
    x_in = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
    x_in_t = torch.tensor(x_in, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(x_in_t).cpu().numpy().flatten()[0]
    forecasts.append(float(pred))
    history.append(float(test_data[t]))
    X_all, y_all = make_sequences(np.array(history, dtype=np.float32), seq_len)
    model = train_model(model, X_all, y_all, epochs, batch_size, lr)

print(forecasts)