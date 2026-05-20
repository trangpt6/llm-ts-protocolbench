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
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values.astype(float)
non_zero = target[target != 0]
median_nonzero = np.median(non_zero)
target = np.where(target == 0, median_nonzero, target)
train_size = 2920
train_data = target[:train_size]
test_data = target[train_size:]
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
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)
def train_model(model, data, seq_len, epochs, batch_size, lr):
    if len(data) <= seq_len:
        return model
    X, y = create_sequences(data, seq_len)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model
input_size = 1
seq_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01
model = GRUModel(input_size, hidden_size, num_layers)
model = train_model(model, train_data, seq_len, epochs, batch_size, lr)
forecasts = []
history = list(train_data)
for i in range(len(test_data)):
    input_seq = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pred = model(input_seq).item()
    forecasts.append(pred)
    history.append(test_data[i])
    model = GRUModel(input_size, hidden_size, num_layers)
    model = train_model(model, np.array(history), seq_len, epochs, batch_size, lr)
print(forecasts)