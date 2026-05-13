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

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
series = df['Passengers'].values.astype(np.float32)

# Chronological split: 80% train, 20% test
train_len = 115
test_len = len(series) - train_len
train_data = series[:train_len]
test_data = series[train_len:]

# Sequence generation
def create_sequences(data, seq_len=12, pred_len=12):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# LSTM model
class LSTMMultiStep(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)            # out: (batch, seq_len, hidden)
        last_out = out[:, -1, :]         # (batch, hidden)
        return self.fc(last_out)         # (batch, pred_len)

# Fixed hyperparameters
input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001

# Block-wise rolling forecast with retraining
history = list(train_data)
forecasts_all = []
device = 'cuda' if torch.cuda.is_available() else 'cpu'

while len(forecasts_all) < test_len:
    # Prepare training sequences from current history
    hist_arr = np.array(history)
    X, y = create_sequences(hist_arr, seq_len, pred_len)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)  # (N,12,1)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)                # (N,12)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = LSTMMultiStep(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Train
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    # Forecast next block
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)  # (1,12,1)
        pred = model(last_seq).cpu().numpy().flatten()  # shape (pred_len,)
    
    # Store predictions, clipping to exact test length
    needed = test_len - len(forecasts_all)
    if needed >= pred_len:
        forecasts_all.extend(pred.tolist())
        # Append true values of this block to history for next retraining
        block_truth = test_data[len(forecasts_all)-pred_len:len(forecasts_all)]
        history.extend(block_truth.tolist())
    else:
        forecasts_all.extend(pred[:needed].tolist())
        break  # No more full blocks needed, no further retraining

# Ensure exactly test_len predictions
forecasts = forecasts_all[:test_len]

print(forecasts)