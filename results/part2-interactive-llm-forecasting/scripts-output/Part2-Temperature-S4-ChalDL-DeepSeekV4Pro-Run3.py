import random
import numpy as np
import torch

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Read and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'], index_col='Date')
series = df['Daily minimum temperatures'].astype(np.float32).values

# Fixed split
train_len = 2921
test_len = len(series) - train_len  # 731

# Hyperparameters
INPUT_SIZE = 1
SEQ_LEN = 14
PRED_LEN = 30
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 20
BATCH_SIZE = 32
LR = 0.001

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# LSTM model definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        out, _ = self.lstm(x)
        # use the last hidden state output for prediction
        last_out = out[:, -1, :]  # (batch, hidden_size)
        pred = self.fc(last_out)  # (batch, pred_len)
        return pred

def create_sequences(data, seq_len, pred_len):
    """Generate sliding windows for training.
       data: 1D numpy array.
       Returns X (n, seq_len, 1) and y (n, pred_len)."""
    X_list, y_list = [], []
    total_len = len(data)
    for i in range(total_len - seq_len - pred_len + 1):
        X_list.append(data[i:i+seq_len].reshape(-1, 1))
        y_list.append(data[i+seq_len:i+seq_len+pred_len])
    if len(X_list) == 0:
        return None, None
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y

def train_model(model, train_loader, epochs, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch_X.size(0)
        # optional: average loss not needed

def predict(model, last_seq):
    """last_seq: numpy array of shape (seq_len,)"""
    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(last_seq).float().view(1, SEQ_LEN, 1).to(device)
        pred = model(inp).cpu().numpy().flatten()
    return pred

# Main multi-step block-wise forecasting
forecasts = []
current_hist_len = train_len  # number of known values (including train)

# All data as full series for convenience
all_data = series.copy()

# Full blocks of 30: 24 blocks (720 days)
num_full_blocks = test_len // PRED_LEN  # 24
for block in range(num_full_blocks):
    # Training data: up to current_hist_len
    train_series = all_data[:current_hist_len]
    X, y = create_sequences(train_series, SEQ_LEN, PRED_LEN)
    if X is None or len(X) < BATCH_SIZE:
        # fallback: use at least one batch; but not expected with this data size
        pass
    train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_LEN).to(device)
    train_model(model, train_loader, EPOCHS, LR)
    
    # Get last window
    last_seq = train_series[-SEQ_LEN:]
    pred_block = predict(model, last_seq)  # shape (30,)
    forecasts.extend(pred_block.tolist())
    
    # Update history with true test values for next block
    current_hist_len += PRED_LEN

# Remaining days (11)
remaining = test_len % PRED_LEN
if remaining > 0:
    # Use all available data up to now (which includes previous blocks' true values)
    train_series = all_data[:current_hist_len]
    X, y = create_sequences(train_series, SEQ_LEN, PRED_LEN)
    train_dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    model = LSTMForecaster(INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, PRED_LEN).to(device)
    train_model(model, train_loader, EPOCHS, LR)
    
    last_seq = train_series[-SEQ_LEN:]
    pred_block = predict(model, last_seq)
    forecasts.extend(pred_block[:remaining].tolist())

# Final output: flat list of forecasts
print(forecasts)