import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df.sort_values('DATE', inplace=True)
df.set_index('DATE', inplace=True)

# Preprocessing: drop columns with excessive missing values
if 'AGE 25-49' in df.columns:
    df.drop(columns=['AGE 25-49'], inplace=True)
if 'AGE 50-64' in df.columns:
    df.drop(columns=['AGE 50-64'], inplace=True)

# Target variable
target = df['% WEIGHTED ILI'].values

# Train/test split
train_size = 1044
train_data = target[:train_size]
test_data = target[train_size:]

# Scaling
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1)).flatten()

# Hyperparameters
seq_len = 52
input_size = 1
hidden_size = 64
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32]
epochs = 50
batch_size = 32
lr = 0.001

# Create sequences
X_train, y_train = [], []
for i in range(len(train_scaled) - seq_len):
    X_train.append(train_scaled[i:i+seq_len])
    y_train.append(train_scaled[i+seq_len])

X_train = torch.tensor(np.array(X_train), dtype=torch.float32).unsqueeze(2)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32).unsqueeze(1)

from torch.utils.data import TensorDataset, DataLoader
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# TCN Model Definition
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0):
        super(TemporalBlock, self).__init__()
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
    def __init__(self, input_size, output_size, hidden_size, dilations, kernel_size):
        super(TCN, self).__init__()
        layers = []
        for i, d in enumerate(dilations):
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=d, padding=(kernel_size-1)*d)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.transpose(1, 2)
        y1 = self.network(x)
        return self.linear(y1[:, :, -1])

model = TCN(input_size=input_size, output_size=1, hidden_size=hidden_size, dilations=dilations, kernel_size=kernel_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
model.train()
for epoch in range(epochs):
    for batch_X, batch_y in loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Inference (Recursive)
model.eval()
forecasts_scaled = []
current_seq = torch.tensor(train_scaled[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)

test_size = len(target) - train_size
with torch.no_grad():
    for _ in range(test_size):
        pred = model(current_seq)
        forecasts_scaled.append(pred.item())
        pred_tensor = pred.view(1, 1, 1)
        current_seq = torch.cat((current_seq[:, 1:, :], pred_tensor), dim=1)

# Inverse transform
forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten().tolist()
print(forecasts)