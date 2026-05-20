import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values
train_size = 158
train_data = data[:train_size]
test_data = data[train_size:]
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last = lstm_out[:, -1, :]
        out = self.fc(last)
        return out
def train_model(data, seq_len, pred_len, input_size, hidden_size, num_layers, epochs, batch_size, lr):
    X = []
    y = []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 1])
    X = np.array(X)
    y = np.array(y)
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model
model = train_model(train_data, 6, 12, 2, 16, 1, 5, 8, 0.01)
forecasts = []
current_data = train_data.copy()
for i in range(len(test_data)):
    window = current_data[-6:]
    window_tensor = torch.tensor(window.reshape(1, 6, 2), dtype=torch.float32)
    with torch.no_grad():
        pred = model(window_tensor).numpy()[0]
    forecasts.append(float(pred[0]))
    true_value = test_data[i]
    current_data = np.vstack([current_data, true_value])
    model = train_model(current_data, 6, 12, 2, 16, 1, 5, 8, 0.01)
print(forecasts)