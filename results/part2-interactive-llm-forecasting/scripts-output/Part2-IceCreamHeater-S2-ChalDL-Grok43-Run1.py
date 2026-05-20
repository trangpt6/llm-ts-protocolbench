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
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = self.fc(gru_out[:, -1, :])
        return out
def train_model(train_data, seq_len, input_size, hidden_size, num_layers, epochs, batch_size, lr):
    model = GRUModel(input_size, hidden_size, num_layers)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X = []
    y = []
    for i in range(len(train_data) - seq_len):
        X.append(train_data[i:i+seq_len])
        y.append(train_data[i+seq_len, 1])
    X = np.array(X)
    y = np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
features = df[['Heater', 'Ice cream']].values.astype(np.float32)
seq_len = 6
input_size = 2
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01
train_size = 158
test_size = 40
current_train_data = features[:train_size].copy()
forecasts = []
for step in range(test_size):
    model = train_model(current_train_data, seq_len, input_size, hidden_size, num_layers, epochs, batch_size, lr)
    last_seq = current_train_data[-seq_len:]
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        pred = model(last_seq_tensor).item()
    forecasts.append(pred)
    next_true = features[train_size + step]
    current_train_data = np.vstack([current_train_data, next_true])
print(forecasts)