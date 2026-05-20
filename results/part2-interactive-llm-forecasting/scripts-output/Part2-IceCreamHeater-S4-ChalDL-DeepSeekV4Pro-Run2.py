import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Chomp layer to maintain correct output length after causal padding
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# Temporal Block with causal dilated convolution
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=1, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu)
    def forward(self, x):
        return self.net(x)

# TCN forecasting model
class TCNForecast(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, seq_len, pred_len):
        super(TCNForecast, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.hidden_size = hidden_size
        self.layers = nn.ModuleList()
        in_channels = input_size
        for i in range(num_layers):
            d = dilations[i]
            out_channels = hidden_size
            self.layers.append(TemporalBlock(in_channels, out_channels, kernel_size, d))
            in_channels = hidden_size
        self.fc = nn.Linear(hidden_size * seq_len, pred_len)
    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        for layer in self.layers:
            x = layer(x)
        x = x.reshape(x.size(0), -1)  # (batch, hidden_size*seq_len)
        return self.fc(x)

# Function to create sliding window samples
def create_windows(data, seq_len, pred_len):
    X_list, y_list = [], []
    max_i = data.shape[0] - seq_len - pred_len
    if max_i < 0:
        return None, None
    for i in range(max_i + 1):
        X_list.append(data[i:i+seq_len])
        y_list.append(data[i+seq_len:i+seq_len+pred_len, 1])  # only Ice cream target
    return np.array(X_list), np.array(y_list)

# Train model on given data array of shape (N, 2)
def train_model(data_array, seq_len=12, pred_len=12, input_size=2, hidden_size=32,
                num_layers=3, kernel_size=3, dilations=[1,2,4,8], epochs=30,
                batch_size=16, lr=0.001):
    X, y = create_windows(data_array, seq_len, pred_len)
    if X is None or len(X) == 0:
        raise ValueError("Not enough data to create training windows.")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TCNForecast(input_size, hidden_size, num_layers, kernel_size,
                        dilations[:num_layers], seq_len, pred_len).to(device)
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Forecast using model and input sequence of shape (seq_len, 2)
def forecast(model, input_seq, steps):
    model.eval()
    device = next(model.parameters()).device
    X_tensor = torch.FloatTensor(input_seq[np.newaxis, :, :]).to(device)  # (1, seq_len, 2)
    with torch.no_grad():
        pred = model(X_tensor)  # (1, pred_len)
    return pred.cpu().numpy().flatten()

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)
train_len = 158  # int(0.8 * 198)
# Initial training
current_data = data[:train_len].copy()
model = train_model(current_data)

# Block-wise rolling forecast
forecasts = []
test_start = train_len
while test_start < len(df):
    if test_start + 12 > len(df):
        needed = len(df) - test_start
    else:
        needed = 12
    # Input is the last seq_len=12 months of current data
    X_input = current_data[-12:]  # (12, 2)
    pred_all = forecast(model, X_input, steps=12)
    pred_needed = pred_all[:needed]
    forecasts.extend(pred_needed.tolist())
    # Obtain ground truth for the forecasted block
    true_block = data[test_start:test_start+needed]
    current_data = np.concatenate([current_data, true_block], axis=0)
    test_start += needed
    # Retrain if more blocks remain
    if test_start < len(df):
        model = train_model(current_data)

print(forecasts)