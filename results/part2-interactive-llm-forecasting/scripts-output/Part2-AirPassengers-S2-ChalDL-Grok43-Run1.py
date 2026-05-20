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
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
def create_sequences(series, seq_len):
    X = []
    y = []
    for i in range(len(series) - seq_len):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len])
    return np.array(X), np.array(y)
forecasts = []
current_train = list(train_data)
for i in range(len(test_data)):
    X_train, y_train = create_sequences(np.array(current_train), 12)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    model = GRUModel(1, 32, 1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    for epoch in range(10):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = torch.tensor(current_train[-12:], dtype=torch.float32).view(1, 12, 1)
    with torch.no_grad():
        pred = model(last_seq).item()
    forecasts.append(pred)
    current_train.append(test_data[i])
print(forecasts)