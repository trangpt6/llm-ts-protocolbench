import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Load data and apply preprocessing from Turn 1 (parse datetime index)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
data = df['Passengers'].values.astype(np.float32)

# Chronological 80/20 split defined in Turn 0
train_size = 115
train = data[:train_size]
test = data[train_size:]

# Create sliding window sequences for LSTM (seq_len=12, input_size=1, pred_len=1)
seq_len = 12
X_train = []
y_train = []
for i in range(len(train) - seq_len):
    X_train.append(train[i:i+seq_len])
    y_train.append(train[i+seq_len])
X_train = np.array(X_train).reshape(-1, seq_len, 1)
y_train = np.array(y_train).reshape(-1, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM model as fixed in Turn 2
class LSTMForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMForecast, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

model = LSTMForecast(input_size=1, hidden_size=64, num_layers=2, pred_len=1).to(device)

# Training
lr = 0.001
epochs = 50
batch_size = 16
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

model.train()
for epoch in range(epochs):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Static recursive one-step-ahead forecasting without ground truth updates
model.eval()
forecasts = []
input_seq = list(train[-seq_len:])
with torch.no_grad():
    for _ in range(len(test)):
        x = torch.tensor(input_seq, dtype=torch.float32).reshape(1, seq_len, 1).to(device)
        pred = model(x).cpu().item()
        forecasts.append(pred)
        input_seq = input_seq[1:] + [pred]

print(forecasts)