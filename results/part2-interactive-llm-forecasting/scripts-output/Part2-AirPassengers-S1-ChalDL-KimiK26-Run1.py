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

# read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
series = df['Passengers'].values.astype(np.float32)

# apply fixed chronological split
train_size = 115
test_size = 29
train_data = series[:train_size]

# create sliding windows for training
seq_len = 12
X_list = []
y_list = []
for i in range(len(train_data) - seq_len):
    X_list.append(train_data[i:i+seq_len])
    y_list.append(train_data[i+seq_len])
X_train = np.array(X_list).reshape(-1, seq_len, 1)
y_train = np.array(y_list).reshape(-1, 1)

# data loader
dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# define LSTM model
class LSTMModel(nn.Module):
    def __init__(self):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMModel()
device = torch.device('cpu')
model = model.to(device)

# training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
model.train()
for epoch in range(50):
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# recursive one-step ahead forecasting over test set
model.eval()
forecasts = []
current_seq = train_data[-seq_len:].reshape(1, seq_len, 1)
with torch.no_grad():
    for _ in range(test_size):
        x = torch.from_numpy(current_seq).to(device)
        pred = model(x).cpu().numpy()[0, 0]
        forecasts.append(float(pred))
        current_seq = np.concatenate([current_seq[:, 1:, :], [[[pred]]]], axis=1)

print(forecasts)