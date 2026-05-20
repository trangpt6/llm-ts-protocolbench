import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values

# Hyperparameters
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

# TCN Components
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
        
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
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
    def __init__(self, input_size, output_size, num_channels, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = dilations[i]
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, padding=padding)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)
        
    def forward(self, x):
        y1 = self.network(x)
        return self.linear(y1[:, :, -1])

# Rolling forecast setup
forecasts = []
train_size = 115
total_size = len(data)

for t in range(train_size, total_size):
    train_data = data[:t]
    
    # MinMax Scaling to prevent exploding gradients
    min_val = np.min(train_data)
    max_val = np.max(train_data)
    train_scaled = (train_data - min_val) / (max_val - min_val + 1e-8)
    
    # Create sequences
    X_train, y_train = [], []
    for i in range(len(train_scaled) - seq_len - pred_len + 1):
        X_train.append(train_scaled[i : i + seq_len])
        y_train.append(train_scaled[i + seq_len : i + seq_len + pred_len])
        
    X_train = torch.tensor(np.array(X_train), dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    
    # Initialize model
    torch.manual_seed(42)
    model = TCN(input_size=input_size, output_size=pred_len, num_channels=[hidden_size]*num_layers, kernel_size=kernel_size, dilations=dilations[:num_layers])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train
    dataset = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()
            
    # Predict
    model.eval()
    with torch.no_grad():
        X_test = torch.tensor(train_scaled[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(1)
        pred_scaled = model(X_test).squeeze(0).numpy()
        pred = pred_scaled * (max_val - min_val + 1e-8) + min_val
        # Append the 1-step ahead forecast from the 12-step horizon
        forecasts.append(float(pred[0]))

print(forecasts)