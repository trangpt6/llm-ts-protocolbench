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
target = df['Daily minimum temperatures'].values.astype(np.float32)
train_size = 2920
train = target[:train_size]
test = target[train_size:]
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
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)
def train_model(model, data, seq_len, epochs, batch_size, lr):
    X, y = create_sequences(data, seq_len)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
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
model = GRUModel(1, 16, 1)
model = train_model(model, train, 7, 5, 16, 0.01)
forecasts = []
current_data = list(train)
for i in range(len(test)):
    input_seq = np.array(current_data[-7:]).reshape(1, 7, 1)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).item()
    forecasts.append(pred)
    current_data.append(test[i])
    model = GRUModel(1, 16, 1)
    model = train_model(model, np.array(current_data), 7, 5, 16, 0.01)
print(forecasts)