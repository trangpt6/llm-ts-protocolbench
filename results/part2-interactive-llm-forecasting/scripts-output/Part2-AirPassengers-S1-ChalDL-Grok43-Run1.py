import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)
train_data = data[:115]
def create_sequences(data, seq_len=12):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i + seq_len])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_data)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
model = LSTMModel(1, 64, 2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
batch_size = 16
epochs = 50
for epoch in range(epochs):
    model.train()
    for i in range(0, len(X_train), batch_size):
        batch_x = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
model.eval()
forecasts = []
current_seq = torch.tensor(train_data[-12:], dtype=torch.float32).view(1, 12, 1)
for _ in range(29):
    with torch.no_grad():
        pred = model(current_seq)
        forecasts.append(float(pred.item()))
        current_seq = torch.cat((current_seq[:, 1:, :], pred.view(1, 1, 1)), dim=1)
print(forecasts)