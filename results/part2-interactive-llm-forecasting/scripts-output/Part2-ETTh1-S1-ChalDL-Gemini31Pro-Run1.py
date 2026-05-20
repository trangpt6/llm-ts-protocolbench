import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import random

# Set random seeds
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Feature engineering to match input_size=7
df['hour'] = df.index.hour.astype(float)
df['dayofweek'] = df.index.dayofweek.astype(float)
df['day'] = df.index.day.astype(float)
df['month'] = df.index.month.astype(float)
df['dayofyear'] = df.index.dayofyear.astype(float)
df['weekofyear'] = df.index.isocalendar().week.astype(float)

features = ['OT', 'hour', 'dayofweek', 'day', 'month', 'dayofyear', 'weekofyear']
data = df[features].values

# Split data
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
train_data = data[:train_size]
test_data = data[train_size:]

# Scale data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)
test_scaled = scaler.transform(test_data)

# Hyperparameters
input_size = 7
seq_len = 48
pred_len = 1
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8, 16]
epochs = 30
batch_size = 64
lr = 0.001

# Create sequences
def create_sequences(data_array, seq_length, pred_length):
    X, y = [], []
    for i in range(len(data_array) - seq_length - pred_length + 1):
        X.append(data_array[i:i+seq_length])
        y.append(data_array[i+seq_length:i+seq_length+pred_length, 0])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, seq_len, pred_len)

train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# TCN Model
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, k_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, k_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, k_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.conv2, self.chomp2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, k_size, dilations_list, n_layers):
        super(TCN, self).__init__()
        layers = []
        for i in range(n_layers):
            dilation_size = dilations_list[i % len(dilations_list)]
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, k_size, stride=1, dilation=dilation_size, padding=(k_size-1) * dilation_size)]
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        return self.network(x)

class TCNModel(nn.Module):
    def __init__(self, in_size, out_size, num_channels, k_size, dilations_list, n_layers):
        super(TCNModel, self).__init__()
        self.tcn = TCN(in_size, num_channels, k_size, dilations_list, n_layers)
        self.linear = nn.Linear(num_channels[-1], out_size)
    def forward(self, x):
        y1 = self.tcn(x.transpose(1, 2))
        return self.linear(y1[:, :, -1])

num_channels_list = [hidden_size] * num_layers
model = TCNModel(input_size, pred_len, num_channels_list, kernel_size, dilations, num_layers).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
model.train()
for epoch in range(epochs):
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Recursive Forecasting
model.eval()
forecasts = []
current_seq = train_scaled[-seq_len:].copy()

test_steps = len(test_data)
with torch.no_grad():
    for i in range(test_steps):
        seq_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(seq_tensor).cpu().numpy()[0]
        
        next_features = test_scaled[i].copy()
        next_features[0] = pred[0]
        
        current_seq = np.vstack((current_seq[1:], next_features))
        
        dummy = np.zeros((1, input_size))
        dummy[0, 0] = pred[0]
        actual_pred = scaler.inverse_transform(dummy)[0, 0]
        forecasts.append(float(actual_pred))

print(forecasts)