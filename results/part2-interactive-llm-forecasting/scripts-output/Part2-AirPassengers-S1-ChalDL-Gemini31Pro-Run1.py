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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)

# Split data
train_size = 115
train_data = data[:train_size]
test_size = len(data) - train_size

# Create sequences
seq_len = 12
X_train = []
y_train = []
for i in range(len(train_data) - seq_len):
    X_train.append(train_data[i:i+seq_len])
    y_train.append(train_data[i+seq_len])

X_train = torch.tensor(np.array(X_train)).unsqueeze(-1).to(device)
y_train = torch.tensor(np.array(y_train)).unsqueeze(-1).to(device)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)

# Define model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, pred_len=1):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(input_size=1, hidden_size=64, num_layers=2, pred_len=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train model
model.train()
for epoch in range(50):
    for batch_X, batch_y in dataloader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# Inference
model.eval()
predictions = []
current_seq = train_data[-seq_len:].tolist()

with torch.no_grad():
    for _ in range(test_size):
        seq_tensor = torch.tensor(current_seq, dtype=torch.float32).view(1, seq_len, 1).to(device)
        pred = model(seq_tensor).item()
        predictions.append(pred)
        current_seq.append(pred)
        current_seq.pop(0)

print(predictions)