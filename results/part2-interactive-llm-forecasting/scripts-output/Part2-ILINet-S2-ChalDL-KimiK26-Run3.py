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

# load dataset
df = pd.read_csv(r'../../../data/ILINet.csv')

# preprocessing from Turn 1
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.drop(columns=['AGE 25-49'])
full_dates = pd.date_range(start=df['DATE'].min(), end=df['DATE'].max(), freq='7D')
df = df.set_index('DATE').reindex(full_dates)
df = df.interpolate(method='linear')
df = df.reset_index().rename(columns={'index': 'DATE'})
df = df.sort_values('DATE').reset_index(drop=True)

# target series
target_col = '% WEIGHTED ILI'
full_series = df[target_col].values.astype(np.float32)

# exact chronological split per Turn 0
split_date = pd.Timestamp('2017-10-08')
split_idx = int((df['DATE'] < split_date).sum())
test_size = 262

# fixed hyperparameters from Turn 2
seq_len = 13
pred_len = 1
input_size = 1
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

device = torch.device('cpu')
forecasts = []

for i in range(test_size):
    train_series = full_series[:split_idx + i]
    X = []
    y = []
    for j in range(len(train_series) - seq_len):
        X.append(train_series[j:j+seq_len])
        y.append(train_series[j+seq_len])
    X = np.array(X).reshape(-1, seq_len, input_size)
    y = np.array(y).reshape(-1, pred_len)
    X_tensor = torch.from_numpy(X).float().to(device)
    y_tensor = torch.from_numpy(y).float().to(device)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRUModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    last_seq = train_series[-seq_len:].reshape(1, seq_len, input_size)
    last_seq_tensor = torch.from_numpy(last_seq).float().to(device)
    with torch.no_grad():
        pred = model(last_seq_tensor).cpu().item()
    forecasts.append(float(pred))

print(forecasts)