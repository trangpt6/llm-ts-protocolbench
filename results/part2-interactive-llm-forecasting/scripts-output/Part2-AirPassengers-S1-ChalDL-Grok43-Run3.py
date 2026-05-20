import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)
train_size = 115
train_data = data[:train_size]
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
input_size = 1
hidden_size = 64
num_layers = 2
seq_len = 12
model = LSTMModel(input_size, hidden_size, num_layers)
X_train = []
y_train = []
for i in range(len(train_data) - seq_len):
    X_train.append(train_data[i:i+seq_len].reshape(-1,1))
    y_train.append(train_data[i+seq_len])
X_train = np.array(X_train)
y_train = np.array(y_train).reshape(-1,1)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 50
batch_size = 16
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
for epoch in range(epochs):
    for Xb, yb in loader:
        optimizer.zero_grad()
        pred = model(Xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
forecasts = []
current_seq = train_data[-seq_len:].reshape(1, seq_len, 1)
current_seq = torch.tensor(current_seq, dtype=torch.float32)
for _ in range(29):
    with torch.no_grad():
        pred = model(current_seq)
    forecasts.append(float(pred.item()))
    current_seq = torch.cat((current_seq[:, 1:, :], pred.unsqueeze(1).unsqueeze(2)), dim=1)
print(forecasts)