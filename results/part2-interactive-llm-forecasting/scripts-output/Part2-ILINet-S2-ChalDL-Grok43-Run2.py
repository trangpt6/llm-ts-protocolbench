import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values.astype(np.float32)
train_size = 1044
train = target[:train_size]
test = target[train_size:]
seq_len = 13
class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(1, 32, 1, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len].reshape(-1, 1))
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
forecasts = []
current_train = list(train)
for i in range(len(test)):
    X_train, y_train = create_sequences(np.array(current_train), seq_len)
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    model = GRUModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model.train()
    for epoch in range(5):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    last_seq = torch.tensor(np.array(current_train[-seq_len:]).reshape(1, seq_len, 1), dtype=torch.float32)
    with torch.no_grad():
        next_pred = model(last_seq).item()
    forecasts.append(next_pred)
    current_train.append(test[i])
print(forecasts)