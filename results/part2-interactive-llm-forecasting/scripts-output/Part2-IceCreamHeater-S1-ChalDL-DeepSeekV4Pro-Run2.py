import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.set_index('Month')
data = df[['Heater', 'Ice cream']]

# Train/test split (chronological, first 158 train)
train_data = data.iloc[:158]
test_data = data.iloc[158:]

# Scale
scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(train_data.values)
scaled_test = scaler.transform(test_data.values)

# Prepare sequences
seq_len = 12
X_train = []
y_train = []
for i in range(len(scaled_train) - seq_len):
    X_train.append(scaled_train[i:i+seq_len])
    y_train.append(scaled_train[i+seq_len, 1])  # Ice cream column index 1
X_train = np.array(X_train)
y_train = np.array(y_train)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TCN model
from torch import nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dilations, dropout=0.0):
        super().__init__()
        layers = []
        num_levels = len(dilations)
        for i in range(num_levels):
            dilation = dilations[i]
            in_channels = input_size if i == 0 else num_channels
            out_channels = num_channels
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation,
                                        padding=(kernel_size-1)*dilation, dropout=dropout))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels, output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = x.permute(0, 2, 1)  # to (batch, features, seq_len)
        out = self.network(x)
        out = out[:, :, -1]  # last time step
        out = self.linear(out)
        return out.squeeze(-1)

input_size = 2
hidden_size = 32
output_size = 1
dilations = [1, 2, 4, 8]
kernel_size = 3
model = TCN(input_size=input_size, output_size=output_size, num_channels=hidden_size,
            kernel_size=kernel_size, dilations=dilations).to(device)

# Training
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
batch_size = 16
epochs = 50

X_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)

model.train()
for epoch in range(epochs):
    permutation = torch.randperm(X_tensor.size(0))
    for i in range(0, X_tensor.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_x = X_tensor[indices]
        batch_y = y_tensor[indices]
        optimizer.zero_grad()
        pred = model(batch_x)
        loss = criterion(pred, batch_y)
        loss.backward()
        optimizer.step()

# Recursive forecasting
model.eval()
forecasts_scaled = []
input_seq = torch.tensor(scaled_train[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)  # shape (1,12,2)
test_len = len(scaled_test)

with torch.no_grad():
    for i in range(test_len):
        next_pred = model(input_seq)  # scalar tensor
        forecasts_scaled.append(next_pred.item())
        # Update sequence: remove oldest, append new point (Heater from test, predicted Ice cream)
        new_point = [scaled_test[i, 0], next_pred.item()]  # heater, ice cream
        new_seq = torch.cat([input_seq[:, 1:, :], torch.tensor(new_point, dtype=torch.float32).to(device).view(1,1,2)], dim=1)
        input_seq = new_seq

# Inverse transform Ice cream
ice_min = scaler.data_min_[1]
ice_max = scaler.data_max_[1]
forecasts = [val * (ice_max - ice_min) + ice_min for val in forecasts_scaled]

print(forecasts)