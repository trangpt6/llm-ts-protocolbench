import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
target = df['% WEIGHTED ILI'].values.astype(float)

# Chronological split (80/20)
total_len = len(target)
train_size = int(0.8 * total_len)  # 1044
test_size = total_len - train_size  # 261
train_data = target[:train_size]
test_data = target[train_size:]

# Scale data using training set only
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_data.reshape(-1, 1)).flatten()

# Hyperparameters (fixed)
seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

# Build sequences for training
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=52):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # take last time step
        out = self.fc(out)
        return out

# Train on given scaled data
def train_model(data_scaled):
    X, y = create_sequences(data_scaled, seq_len, pred_len)
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # shape (samples, seq_len, 1)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMModel(input_size=1, hidden_size=hidden_size, num_layers=num_layers, output_size=pred_len)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Rolling block forecast
forecasts = []
current_train_scaled = train_scaled.copy()

for block in range(6):
    # Train model on current training data
    model = train_model(current_train_scaled)
    model.eval()
    # last seq_len values of current training
    last_seq = current_train_scaled[-seq_len:]
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).view(1, seq_len, 1)
    with torch.no_grad():
        block_pred_scaled = model(last_seq_tensor).numpy().flatten()
    # Inverse transform to original scale
    block_pred = scaler.inverse_transform(block_pred_scaled.reshape(-1, 1)).flatten()
    # Determine how many of these predictions are actually needed for test set
    remaining = test_size - len(forecasts)
    take = min(pred_len, remaining)
    forecasts.extend(block_pred[:take].tolist())
    # If we have enough forecasts, break
    if len(forecasts) >= test_size:
        break
    # Update training data with true test values for this block
    start_idx = len(forecasts) - take  # index in test set where this block started
    end_idx = start_idx + take
    true_test_scaled = test_scaled[start_idx:end_idx]
    current_train_scaled = np.concatenate([current_train_scaled, true_test_scaled])

print(forecasts)