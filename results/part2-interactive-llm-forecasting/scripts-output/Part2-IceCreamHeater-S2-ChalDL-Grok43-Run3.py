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
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)
train_size = 158
seq_len = 6
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
def train_model(model, train_data, seq_len, epochs, batch_size, lr):
    X = []
    y = []
    for i in range(len(train_data) - seq_len):
        X.append(train_data[i:i+seq_len])
        y.append(train_data[i + seq_len, 1])
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model
forecasts = []
current_train = data[:train_size].copy()
for i in range(40):
    model = GRUModel(2, 16, 1)
    model = train_model(model, current_train, seq_len, 5, 8, 0.01)
    last_seq = torch.tensor(current_train[-seq_len:], dtype=torch.float32).unsqueeze(0)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).item()
    forecasts.append(pred)
    next_row = data[train_size + i]
    current_train = np.vstack((current_train, next_row))
print(forecasts)