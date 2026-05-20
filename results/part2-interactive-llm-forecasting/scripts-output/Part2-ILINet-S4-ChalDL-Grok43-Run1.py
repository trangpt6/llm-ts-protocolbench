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
seq_len = 52
pred_len = 52
input_size = 1
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001
device = torch.device('cpu')
class LSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])
model = LSTM().to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train, seq_len, pred_len)
X_train = torch.tensor(X_train).unsqueeze(-1).to(device)
y_train = torch.tensor(y_train).to(device)
train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size, shuffle=True)
for epoch in range(epochs):
    model.train()
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
forecasts = []
current_block = train[-seq_len:].tolist()
for i in range(0, len(test), pred_len):
    model.eval()
    with torch.no_grad():
        inp = torch.tensor(current_block[-seq_len:]).unsqueeze(0).unsqueeze(-1).float().to(device)
        pred = model(inp).cpu().numpy().flatten()
    block_len = min(pred_len, len(test) - i)
    forecasts.extend(pred[:block_len])
    true_block = test[i:i+block_len].tolist()
    current_block.extend(true_block)
    Xb, yb = create_sequences(np.array(current_block), seq_len, pred_len)
    if len(Xb) > 0:
        Xb = torch.tensor(Xb).unsqueeze(-1).float().to(device)
        yb = torch.tensor(yb).to(device)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            p = model(Xb[:1])
            loss = criterion(p, yb[:1])
            loss.backward()
            optimizer.step()
print(forecasts)