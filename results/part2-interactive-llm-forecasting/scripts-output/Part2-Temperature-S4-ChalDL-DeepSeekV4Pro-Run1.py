import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
series = df['Daily minimum temperatures'].values.astype(np.float32)

# Train/test split indices
train_size = 2921
test_size = len(series) - train_size  # 731

# LSTM model definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, pred_len):
        super(LSTMForecaster, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(x)
        # Use last time step output for prediction
        out = self.fc(lstm_out[:, -1, :])
        return out  # (batch, pred_len)

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

def train_model(model, X_train, y_train, epochs, batch_size, lr, device):
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Hyperparameters
input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001
block_size = 30
forecast_horizon = 30

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Prepare initial training data
train_data_initial = series[:train_size].reshape(-1, 1)
scaler = MinMaxScaler()
scaled_train_initial = scaler.fit_transform(train_data_initial).flatten()

# Create sequences for initial training
X_train, y_train = create_sequences(scaled_train_initial, seq_len, pred_len)
X_train = X_train.reshape(-1, seq_len, input_size)

# Initialize and train model
model = LSTMForecaster(input_size, hidden_size, num_layers, seq_len, pred_len).to(device)
model = train_model(model, X_train, y_train, epochs, batch_size, lr, device)

# Prepare for forecasting
forecasts = []
current_data = scaled_train_initial.copy()  # will be extended with true test values

# Forecasting loop over blocks
remaining = test_size
start_idx = 0
while remaining > 0:
    steps_to_forecast = min(block_size, remaining)
    # Use the last seq_len points from current_data as input
    input_seq = current_data[-seq_len:].reshape(1, seq_len, input_size)
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
        pred_scaled = model(input_tensor).cpu().numpy().flatten()
    # Take only the required number of predictions for this block
    pred_block_scaled = pred_scaled[:steps_to_forecast]
    # Inverse transform to original scale
    # Note: scaler was fit on initial training data only; we should refit after retraining? 
    # We'll refit scaler on the entire current_data (which includes true values) after updating, for next blocks.
    # For now, inverse transform using the existing scaler.
    pred_block = scaler.inverse_transform(pred_block_scaled.reshape(-1, 1)).flatten()
    forecasts.extend(pred_block.tolist())
    
    # Append true values from test set to current_data and retrain
    true_block = series[train_size + start_idx : train_size + start_idx + steps_to_forecast]
    # Update scaler: refit on the entire current data (including true block) to be safe
    extended_data = np.concatenate([current_data, (true_block.reshape(-1, 1) - scaler.min_[0]) / scaler.scale_[0] 
                                   if scaler.scale_[0] > 0 else true_block.flatten()])
    # Actually easier: rescale the whole series (train + true blocks so far) using a new scaler
    # We'll just append the true values (original scale) and refit scaler each block.
    current_data = np.concatenate([current_data, (true_block.reshape(-1, 1) - scaler.min_[0]) / scaler.scale_[0] 
                                   if scaler.scale_[0] > 0 else true_block.flatten()])
    # Re-fit scaler on the entire current_data (scaled version) to keep consistency after retraining?
    # Better approach: We'll maintain the original series values and re-apply scaling from scratch each retrain.
    # Simplify: we'll maintain original values in `full_data_so_far` and re-scale when training.
    # I'll adjust to avoid confusion.

# The above code snippet is incomplete and may contain bugs; I'll write a clean script from scratch.

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Read dataset
df = pd.read_csv(r'../../../data/Temperature.csv', parse_dates=['Date'])
df.sort_values('Date', inplace=True)
series = df['Daily minimum temperatures'].values.astype(np.float32)

# Train/test split indices
train_size = 2921
test_size = len(series) - train_size

# LSTM model definition
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, seq_len, pred_len):
        super(LSTMForecaster, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

def train_model(model, X_train, y_train, epochs, batch_size, lr, device):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train, dtype=torch.float32).to(device),
        torch.tensor(y_train, dtype=torch.float32).to(device)
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Fixed hyperparameters
input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001
block_size = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initial training data
train_data = series[:train_size].reshape(-1, 1)
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data).flatten()
X_train, y_train = create_sequences(scaled_train, seq_len, pred_len)
X_train = X_train.reshape(-1, seq_len, input_size)

# Initialize model and train
model = LSTMForecaster(input_size, hidden_size, num_layers, seq_len, pred_len).to(device)
model = train_model(model, X_train, y_train, epochs, batch_size, lr, device)

# Forecasting
forecasts = []
full_data = series[:train_size].copy()  # original values up to train end
current_scaled = scaled_train.copy()

test_start_idx = train_size
remaining = test_size
while remaining > 0:
    steps = min(block_size, remaining)
    # Prepare input: last seq_len points from current_scaled
    input_seq = current_scaled[-seq_len:].reshape(1, seq_len, input_size)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()[:steps]
    # Inverse transform
    pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).flatten()
    forecasts.extend(pred.tolist())

    # Append true test values for this block to full_data
    true_block = series[test_start_idx : test_start_idx + steps]
    full_data = np.concatenate([full_data, true_block])

    # Update scaler on the new full_data and rescale
    scaler = MinMaxScaler()
    scaled_full = scaler.fit_transform(full_data.reshape(-1, 1)).flatten()
    current_scaled = scaled_full

    # Retrain model on the expanded full_data
    if remaining - steps > 0:  # no need to retrain after last block
        X_new, y_new = create_sequences(current_scaled, seq_len, pred_len)
        X_new = X_new.reshape(-1, seq_len, input_size)
        model = LSTMForecaster(input_size, hidden_size, num_layers, seq_len, pred_len).to(device)
        model = train_model(model, X_new, y_new, epochs, batch_size, lr, device)

    test_start_idx += steps
    remaining -= steps

print(forecasts)