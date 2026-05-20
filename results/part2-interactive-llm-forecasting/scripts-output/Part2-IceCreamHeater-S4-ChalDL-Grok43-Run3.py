import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df.set_index('Month', inplace=True)
features = ['Heater', 'Ice cream']
train_size = 158
train_data = df[features].iloc[:train_size].values
test_data = df[features].iloc[train_size:].values
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1, self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super(TCN, self).__init__()
        layers = []
        num_channels = [hidden_size] * num_layers
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 2**i
            padding = (kernel_size - 1) * dilation
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation, padding=padding, dropout=0.2)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], pred_len)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x)
        y = y[:, :, -1]
        y = self.linear(y)
        return y
input_size = 2
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 30
batch_size = 16
lr = 0.001
model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, -1])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_data, seq_len, pred_len)
if len(X_train) > 0:
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
forecasts = []
current_data = train_data.copy()
test_index = 0
while test_index < len(test_data):
    if len(current_data) >= seq_len:
        input_seq = current_data[-seq_len:]
        input_tensor = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            pred = model(input_tensor)
        pred = pred.squeeze().numpy()
        remaining = len(test_data) - test_index
        steps = min(pred_len, remaining)
        forecasts.extend(pred[:steps].tolist())
        true_block = test_data[test_index : test_index + steps]
        current_data = np.vstack([current_data, true_block])
        test_index += steps
        X_new, y_new = create_sequences(current_data, seq_len, pred_len)
        if len(X_new) > 0:
            X_new = torch.tensor(X_new, dtype=torch.float32)
            y_new = torch.tensor(y_new, dtype=torch.float32)
            dataset = TensorDataset(X_new, y_new)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
            for epoch in range(epochs):
                for batch_x, batch_y in loader:
                    optimizer.zero_grad()
                    output = model(batch_x)
                    loss = criterion(output, batch_y)
                    loss.backward()
                    optimizer.step()
print(forecasts)