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
# Device fallback to CPU
device = torch.device('cpu')
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
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
def train_model(train_data, seq_len, input_size, hidden_size, num_layers, epochs, batch_size, lr):
    if len(train_data) <= seq_len:
        return None
    X, y = create_sequences(train_data, seq_len)
    X = X.reshape(-1, seq_len, input_size)
    y = y.reshape(-1, 1)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRUModel(input_size, hidden_size, num_layers).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model
forecasts = []
current_train = train_data.copy()
for i in range(len(test_data)):
    model = train_model(current_train, seq_len, input_size, hidden_size, num_layers, epochs, batch_size, lr)
    if model is None:
        pred = current_train[-1]
    else:
        last_seq = current_train[-seq_len:].reshape(1, seq_len, input_size)
        last_seq = torch.tensor(last_seq, dtype=torch.float32).to(device)
        model.eval()
        with torch.no_grad():
            pred = model(last_seq).item()
    forecasts.append(pred)
    current_train = np.append(current_train, test_data[i])
print(forecasts)