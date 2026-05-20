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

# Device configuration
device = torch.device('cpu')

# Load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
data = df['Daily minimum temperatures'].values

# Train/test split
train_size = 2920
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
input_size = 1
seq_len = 14
pred_len = 1
hidden_size = 64
num_layers = 3
epochs = 50
batch_size = 32
lr = 0.001
kernel_size = 3
dilations = [1, 2, 4, 8][:num_layers]

# Prepare training data
X_train = []
y_train = []
for i in range(len(train_data) - seq_len):
    X_train.append(train_data[i:i+seq_len])
    y_train.append(train_data[i+seq_len])

X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1).to(device)

dataset = torch.utils.data.TensorDataset(X_train, y_train)
loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define TCN Model
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
                                 self.conv2, self.chomp2, self.relu2)
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

# Initialize model, optimizer, and loss function
num_channels = [hidden_size] * num_layers
model = TCN(input_size, pred_len, num_channels, kernel_size, dilations).to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Train the model
model.train()
for epoch in range(epochs):
    for batch_X, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Recursive forecasting
model.eval()
forecasts = []
current_seq = train_data[-seq_len:].tolist()

with torch.no_grad():
    for _ in range(len(test_data)):
        x_input = torch.tensor(current_seq, dtype=torch.float32).view(1, 1, seq_len).to(device)
        pred = model(x_input).item()
        forecasts.append(pred)
        current_seq.append(pred)
        current_seq.pop(0)

print(forecasts)