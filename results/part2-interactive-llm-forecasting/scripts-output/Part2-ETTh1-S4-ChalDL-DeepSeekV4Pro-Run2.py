import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and parse dates
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df = df.set_index('date').sort_index()
target = 'OT'

# Detect and replace constant days (all 24 hourly values identical)
def replace_constant_days(series):
    s = series.copy()
    # group by date (floor)
    daily_groups = s.resample('D')
    for day, group in daily_groups:
        if len(group) < 24: # skip incomplete days
            continue
        if group.nunique() == 1:
            s.loc[group.index] = np.nan
    return s

series = df[target].copy()
series = replace_constant_days(series)
series = series.interpolate(method='linear')

# Chronological split
n = len(series)
train_size = int(0.8 * n)  # 14016
train_series = series.iloc[:train_size]
test_series = series.iloc[train_size:]

# Scale values using initial training set
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(train_series.values.reshape(-1, 1))
train_scaled = scaler.transform(train_series.values.reshape(-1, 1)).flatten()
test_scaled = scaler.transform(test_series.values.reshape(-1, 1)).flatten()

# Hyperparameters
seq_len = 168
pred_len = 168
input_size = 7
hidden_size = 64
num_layers = 4
kernel_size = 4
dilations = [1, 2, 4, 8]
epochs = 20
batch_size = 32
lr = 0.001

# Build feature series with 7 channels (t, t-1, ..., t-6)
def make_feature_series(scaled_array):
    n = len(scaled_array)
    features = np.full((n, input_size), np.nan)
    for i in range(n):
        vals = []
        for lag in range(input_size):
            idx = i - lag
            if idx >= 0:
                vals.append(scaled_array[idx])
            else:
                vals.append(0.0)  # pad with 0
        features[i] = vals
    return features

# Generate sequences for training
def create_sequences(scaled_array):
    feat = make_feature_series(scaled_array)
    X, y = [], []
    max_start = len(feat) - seq_len - pred_len
    for i in range(0, max_start, 1):
        X.append(feat[i:i+seq_len])
        y.append(scaled_array[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

# Define TCN components
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
    def __init__(self, num_inputs, num_channels, kernel_size, dilations):
        super(TemporalConvNet, self).__init__()
        layers = []
        in_channels = num_inputs
        for i, out_channels in enumerate(num_channels):
            dilation = dilations[i] if i < len(dilations) else 1
            padding = (kernel_size - 1) * dilation
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation, padding=padding))
            in_channels = out_channels
        self.tcn = nn.Sequential(*layers)
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len) for Conv1d
        y = self.tcn(x)
        return y.transpose(1, 2)  # back to (batch, seq_len, hidden_size)

class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super(TCNModel, self).__init__()
        self.tcn = TemporalConvNet(input_size, [hidden_size]*num_layers, kernel_size, dilations)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        tcn_out = self.tcn(x)  # (batch, seq_len, hidden_size)
        # take output at the last time step
        last_out = tcn_out[:, -1, :]
        return self.fc(last_out)

def train_model(train_scaled, seq_len, pred_len, epochs, batch_size, lr):
    X, y = create_sequences(train_scaled)
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TCNModel(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    return model

# Forecasting block-wise with retraining
forecast_scaled = []
cumulative_train = train_scaled.copy()  # will expand with true test values
current_pos = 0  # index within test_scaled
full_test_len = len(test_scaled)

while current_pos < full_test_len:
    # Train model on current cumulative training set
    model = train_model(cumulative_train, seq_len, pred_len, epochs, batch_size, lr)
    # Prepare input: last seq_len points of cumulative_train as features
    feat_series = make_feature_series(cumulative_train)
    input_seq = feat_series[-seq_len:]  # shape (seq_len, input_size)
    input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, seq_len, input_size)
    model.eval()
    with torch.no_grad():
        pred = model(input_tensor).cpu().numpy().flatten()  # (pred_len,)
    # Determine how many predictions to take in this block
    remaining = full_test_len - current_pos
    take = min(pred_len, remaining)
    forecast_scaled.extend(pred[:take])
    # Append true test values to cumulative train (ground truth)
    true_block = test_scaled[current_pos:current_pos+take]
    cumulative_train = np.concatenate([cumulative_train, true_block])
    current_pos += take

# Inverse scale
forecast = scaler.inverse_transform(np.array(forecast_scaled).reshape(-1, 1)).flatten().tolist()
print(forecast)