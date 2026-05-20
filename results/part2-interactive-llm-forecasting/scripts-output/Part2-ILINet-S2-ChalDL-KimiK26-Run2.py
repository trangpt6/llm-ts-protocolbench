import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# fallback to cpu if cuda is unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv')
df = df.replace(r'^\s*$', np.nan, regex=True)
target = pd.to_numeric(df['% WEIGHTED ILI'], errors='coerce').values
if np.isnan(target).any():
    target = pd.Series(target).ffill().bfill().values

train_size = 1044
history = target[:train_size].tolist()
forecasts = []

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size=1, hidden_size=32, num_layers=1, batch_first=True)
        self.fc = nn.Linear(32, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

seq_len = 13
for i in range(len(target) - train_size):
    train_series = np.array(history)
    X = []
    y = []
    for j in range(len(train_series) - seq_len):
        X.append(train_series[j:j+seq_len])
        y.append(train_series[j+seq_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y).reshape(-1, 1)
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = GRUModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    model.train()
    for epoch in range(5):
        for batch_x, batch_y in dataloader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        x_input = torch.tensor(train_series[-seq_len:].reshape(1, seq_len, 1), dtype=torch.float32, device=device)
        pred = model(x_input).cpu().numpy()[0, 0]
    forecasts.append(float(pred))
    history.append(float(target[train_size + i]))

print(forecasts)