import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Path to the CSV file
CSV_PATH = 'input.csv'

# Reproducibility seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Read data
df = pd.read_csv(CSV_PATH)
series = df['Passengers'].values.astype(np.float32)

# Chronological split
total_len = len(series)
train_len = int(0.8 * total_len)  # 115
train_data = series[:train_len]
test_data = series[train_len:]

# Fixed hyperparameters
input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 2
epochs = 10
batch_size = 8
lr = 0.01
kernel_size = 3
dilations = [1, 2]  # use first two dilations for two layers

# Define TCN module
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               dilation=dilation,
                               padding=(kernel_size-1)*dilation)
        self.relu = nn.ReLU()
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        res = x if self.downsample is None else self.downsample(x)
        return out + res

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_len, kernel_size,
                 num_layers, dilations):
        super(TCN, self).__init__()
        self.initial_conv = nn.Conv1d(input_size, hidden_size, kernel_size,
                                      padding=kernel_size-1)
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(
                TemporalBlock(hidden_size, hidden_size, kernel_size, dilations[i])
            )
        self.output_conv = nn.Conv1d(hidden_size, output_len, kernel_size=1)

    def forward(self, x):
        # x shape: (batch, input_size, seq_len)
        x = self.initial_conv(x)
        for block in self.blocks:
            x = block(x)
        out = self.output_conv(x)  # (batch, output_len, seq_len)
        out = out[:, :, -1]  # take last time step
        return out

# Scaling utility (min-max)
def create_scaler(series_vals):
    min_val = np.min(series_vals)
    max_val = np.max(series_vals)
    def scale(s): return (s - min_val) / (max_val - min_val + 1e-8)
    def inverse_scale(s):
        return s * (max_val - min_val + 1e-8) + min_val
    return scale, inverse_scale

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# Rolling forecast loop
full_history = train_data.copy()
forecasts = []

for step in range(len(test_data)):
    # Fit scaler on current history only
    scale, inv_scale = create_scaler(full_history)
    scaled_history = scale(full_history)
    # Create training sequences
    X_train, y_train = create_sequences(scaled_history, seq_len, pred_len)
    # Convert to tensors
    X_train_t = torch.tensor(X_train, dtype=torch.float32).reshape(-1, input_size, seq_len).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Initialize model
    model = TCN(input_size, hidden_size, pred_len, kernel_size, num_layers, dilations).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train for fixed epochs
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()

    # Prepare input for forecast (last seq_len points of current scaled history)
    last_seq = scaled_history[-seq_len:]
    x_input = torch.tensor(last_seq, dtype=torch.float32).reshape(1, input_size, seq_len).to(device)
    model.eval()
    with torch.no_grad():
        scaled_forecast = model(x_input).cpu().numpy().flatten()  # shape (pred_len,)
    forecast_values = inv_scale(scaled_forecast)
    # We only need the first prediction (one-step ahead)
    next_pred = forecast_values[0]
    forecasts.append(next_pred)

    # Update history with ground truth from test set
    true_next = test_data[step]
    full_history = np.append(full_history, true_next)

# Output final forecasts as flat list
print(forecasts)