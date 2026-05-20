import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import random
# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].astype(float).values
for i in range(len(target)):
  if target[i] == 0 and i >= 7:
    target[i] = np.median(target[i-7:i])
train_size = 2920
train_data = target[:train_size].tolist()
test_data = target[train_size:].tolist()
class LSTMModel(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers, pred_len):
    super(LSTMModel, self).__init__()
    self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    self.fc = nn.Linear(hidden_size, pred_len)
  def forward(self, x):
    out, _ = self.lstm(x)
    out = out[:, -1, :]
    out = self.fc(out)
    return out
def prepare_sequences(series, seq_len, pred_len):
  X = []
  y = []
  for i in range(len(series) - seq_len - pred_len + 1):
    X.append(series[i:i+seq_len])
    y.append(series[i+seq_len:i+seq_len+pred_len])
  return np.array(X), np.array(y)
def train_model(series, seq_len=14, pred_len=30, hidden_size=64, num_layers=2, epochs=20, batch_size=32, lr=0.001):
  X, y = prepare_sequences(series, seq_len, pred_len)
  if len(X) == 0:
    return None
  X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
  y = torch.tensor(y, dtype=torch.float32)
  model = LSTMModel(1, hidden_size, num_layers, pred_len)
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)
  criterion = nn.MSELoss()
  dataset = TensorDataset(X, y)
  loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  for epoch in range(epochs):
    model.train()
    for batch_X, batch_y in loader:
      optimizer.zero_grad()
      outputs = model(batch_X)
      loss = criterion(outputs, batch_y)
      loss.backward()
      optimizer.step()
  return model
forecasts = []
current_train = train_data.copy()
test_idx = 0
while test_idx < len(test_data):
  model = train_model(current_train)
  if model is None:
    break
  last_seq = current_train[-14:]
  last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
  model.eval()
  with torch.no_grad():
    pred = model(last_seq_tensor).squeeze().tolist()
  remaining = len(test_data) - test_idx
  steps = min(30, remaining)
  forecast_block = pred[:steps]
  forecasts.extend(forecast_block)
  true_block = test_data[test_idx:test_idx+steps]
  current_train.extend(true_block)
  test_idx += steps
print(forecasts)