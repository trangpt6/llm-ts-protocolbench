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

# Hyperparameters from Turn 2
input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 2
epochs = 10
batch_size = 8
lr = 0.01
kernel_size = 3
dilations = [1, 2, 4, 8]

# TCN module definition
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dilations=None, dropout=0.0):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels) - 1
        for i in range(num_levels):
            dilation_size = dilations[i] if dilations else 2**i
            in_channels = num_inputs if i == 0 else num_channels[i]
            out_channels = num_channels[i+1]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size,
                                     dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TCNForecast(nn.Module):
    def __init__(self, input_size, seq_len, pred_len, hidden_size, num_layers, kernel_size, dilations):
        super(TCNForecast, self).__init__()
        self.seq_len = seq_len
        # Build channel list: input_size -> hidden_size repeated num_layers times
        num_channels = [input_size] + [hidden_size] * num_layers
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dilations)
        self.fc = nn.Linear(hidden_size * seq_len, pred_len * input_size)
        self.pred_len = pred_len
        self.input_size = input_size

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> need (batch, input_size, seq_len)
        x = x.permute(0, 2, 1)
        out = self.tcn(x)  # (batch, hidden_size, seq_len)
        out = out.reshape(out.size(0), -1)  # flatten
        out = self.fc(out)  # (batch, pred_len*input_size)
        out = out.view(-1, self.pred_len, self.input_size)  # (batch, pred_len, input_size)
        return out

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv', parse_dates=['Month'])
series = df['Passengers'].values.astype(np.float32)

# Train/test split
n = len(series)
train_size = int(0.8 * n)  # 115
test_size = n - train_size  # 29

# Normalize using only training data for stability
train_mean = series[:train_size].mean()
train_std = series[:train_size].std()
if train_std == 0:
    train_std = 1
series_norm = (series - train_mean) / train_std

# Prepare rolling forecasting
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
forecasts = []

for t in range(test_size):
    # Current training data: indices 0 to train_size + t - 1
    current_train_end = train_size + t
    train_data = series_norm[:current_train_end]
    
    # Build sequences: input seq_len, output pred_len
    X_list, y_list = [], []
    for i in range(len(train_data) - seq_len - pred_len + 1):
        X_list.append(train_data[i:i+seq_len])
        y_list.append(train_data[i+seq_len:i+seq_len+pred_len])
    
    if len(X_list) == 0:
        # fallback: use at least one sample, but with 115+ points won't happen
        break
    
    X = np.array(X_list).reshape(-1, seq_len, 1)
    y = np.array(y_list).reshape(-1, pred_len, 1)
    
    X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
    y_tensor = torch.tensor(y, dtype=torch.float32, device=device)
    
    # Instantiate model
    model = TCNForecast(input_size, seq_len, pred_len, hidden_size, num_layers, kernel_size, dilations)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        permutation = torch.randperm(len(X_tensor))
        for i in range(0, len(X_tensor), batch_size):
            indices = permutation[i:i+batch_size]
            batch_x = X_tensor[indices]
            batch_y = y_tensor[indices]
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
    
    # Forecast next step
    model.eval()
    # Use last seq_len from training data
    last_seq = train_data[-seq_len:].reshape(1, seq_len, 1)
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32, device=device)
    with torch.no_grad():
        pred = model(last_seq_tensor)  # (1, pred_len, 1)
    pred_norm = pred[0, 0, 0].item()  # first step forecast
    # Denormalize
    forecast_value = pred_norm * train_std + train_mean
    forecasts.append(forecast_value)

# Output final forecast list as Python list
print(forecasts)