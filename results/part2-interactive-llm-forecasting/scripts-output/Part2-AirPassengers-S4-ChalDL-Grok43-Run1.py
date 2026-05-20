import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
# Device fallback to CPU if CUDA unavailable
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda')
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTM, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out
def prepare_sequences(data, seq_len, pred_len):
    X = []
    y = []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(float).tolist()
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]
current_data = train_data.copy()
forecasts = []
test_index = 0
seq_len = 12
pred_len = 12
while test_index < len(test_data):
    X_list, y_list = prepare_sequences(current_data, seq_len, pred_len)
    if len(X_list) == 0:
        break
    X_train = torch.tensor(X_list, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_list, dtype=torch.float32)
    model = LSTM(1, 64, 2, 12).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=12, shuffle=True)
    model.train()
    for epoch in range(30):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = np.array(current_data[-seq_len:]).reshape(1, seq_len, 1)
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq_t).cpu().numpy().flatten()
    steps_this_block = min(pred_len, len(test_data) - test_index)
    forecasts.extend(pred[:steps_this_block].tolist())
    for i in range(steps_this_block):
        current_data.append(test_data[test_index + i])
    test_index += steps_this_block
print(forecasts)