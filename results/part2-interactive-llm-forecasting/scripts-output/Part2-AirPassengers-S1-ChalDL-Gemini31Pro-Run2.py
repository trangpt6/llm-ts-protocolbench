import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)

# Train and test split
train_size = 115
test_size = len(data) - train_size
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 64
num_layers = 2
epochs = 50
batch_size = 16
lr = 0.001

# Prepare training sequences
X_train = []
y_train = []
for i in range(len(train_data) - seq_len):
    X_train.append(train_data[i:i+seq_len])
    y_train.append(train_data[i+seq_len])

X_train = torch.tensor(np.array(X_train)).unsqueeze(-1).to(device)
y_train = torch.tensor(np.array(y_train)).unsqueeze(-1).to(device)

# Define LSTM Model
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMForecast(input_size, hidden_size, num_layers, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
model.train()
num_batches = int(np.ceil(len(X_train) / batch_size))
for epoch in range(epochs):
    for i in range(num_batches):
        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, len(X_train))
        batch_X = X_train[start_idx:end_idx]
        batch_y = y_train[start_idx:end_idx]
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Recursive forecasting
model.eval()
forecasts = []
current_seq = train_data[-seq_len:].tolist()

with torch.no_grad():
    for _ in range(test_size):
        seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(seq_tensor).item()
        forecasts.append(pred)
        current_seq.append(pred)
        current_seq.pop(0)

# Print final forecast list
print(forecasts)