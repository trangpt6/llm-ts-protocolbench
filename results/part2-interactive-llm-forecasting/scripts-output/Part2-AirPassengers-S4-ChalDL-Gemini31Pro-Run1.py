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

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)

# Split data
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001

# Model definition
class LSTMForecaster(nn.Module):
    def __init__(self):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Feature engineering
def create_sequences(ts, seq_length, pred_length):
    X, y = [], []
    for i in range(len(ts) - seq_length - pred_length + 1):
        X.append(ts[i:i+seq_length])
        y.append(ts[i+seq_length:i+seq_length+pred_length])
    if len(X) == 0:
        return torch.empty(0, seq_length, 1), torch.empty(0, pred_length)
    return torch.tensor(np.array(X)).unsqueeze(-1), torch.tensor(np.array(y))

# Training function
def train_model(ts):
    X_train, y_train = create_sequences(ts, seq_len, pred_len)
    model = LSTMForecaster().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if len(X_train) == 0:
        return model
        
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
    return model

# Rolling forecast
forecasts = []
current_train = list(train_data)
test_idx = 0

while test_idx < len(test_data):
    # Train model on current history
    model = train_model(np.array(current_train, dtype=np.float32))
    model.eval()
    
    # Prepare input sequence
    last_seq = np.array(current_train[-seq_len:], dtype=np.float32)
    last_seq_tensor = torch.tensor(last_seq).unsqueeze(0).unsqueeze(-1).to(device)
    
    # Predict
    with torch.no_grad():
        pred = model(last_seq_tensor).cpu().numpy()[0]
        
    # Determine how many steps to take from this block
    steps_to_take = min(pred_len, len(test_data) - test_idx)
    forecasts.extend(pred[:steps_to_take].tolist())
    
    # Update training set with ground truth
    current_train.extend(test_data[test_idx:test_idx+steps_to_take])
    test_idx += steps_to_take

print(forecasts)