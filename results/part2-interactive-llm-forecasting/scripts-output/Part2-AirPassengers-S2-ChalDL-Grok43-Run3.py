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
data = df['Passengers'].values.astype(float)
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
input_size = 1
seq_len = 12
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
lr = 0.01
forecasts = []
current_train = train_data.copy()
for i in range(len(test_data)):
    X_train, y_train = create_sequences(current_train, seq_len)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    model = GRUModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = current_train[-seq_len:]
    last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).item()
    forecasts.append(pred)
    current_train = np.append(current_train, test_data[i])
print(forecasts)