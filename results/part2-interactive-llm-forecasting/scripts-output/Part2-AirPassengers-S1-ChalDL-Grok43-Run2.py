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
train_data = data[:115]
def create_sequences(data, seq_len):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_data, 12)
X_train = torch.tensor(X_train).unsqueeze(-1)
y_train = torch.tensor(y_train).unsqueeze(-1)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
model = LSTMModel(1, 64, 2)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=16, shuffle=True)
for epoch in range(50):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
current_seq = train_data[-12:].tolist()
forecasts = []
for _ in range(29):
    input_seq = torch.tensor(current_seq).float().unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(input_seq).item()
    forecasts.append(pred)
    current_seq = current_seq[1:] + [pred]
print(forecasts)