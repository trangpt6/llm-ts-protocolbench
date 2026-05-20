import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Define TCN Model Structure
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.utils.weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()
    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = num_channels[i]
            in_channels = input_size if i == 0 else 32
            out_channels = 32
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(32, output_size)
    def forward(self, x):
        y1 = self.network(x)
        return self.linear(y1[:, :, -1])

# Load data
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(float)

# Setup parameters
train_size = 115
test_size = 29
seq_len = 12
pred_len = 12
epochs = 10
batch_size = 8
lr = 0.01
dilations = [1, 2, 4, 8]
kernel_size = 3

def train_model(train_data):
    x_train, y_train = [], []
    for i in range(len(train_data) - seq_len - pred_len + 1):
        x_train.append(train_data[i:i+seq_len])
        y_train.append(train_data[i+seq_len:i+seq_len+pred_len])
    x_train = torch.tensor(np.array(x_train), dtype=torch.float32).unsqueeze(1)
    y_train = torch.tensor(np.array(y_train), dtype=torch.float32)
    
    model = TCN(1, pred_len, dilations, kernel_size, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(epochs):
        for i in range(0, len(x_train), batch_size):
            batch_x = x_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    return model

# Rolling forecast with retraining
all_forecasts = []
current_history = list(data[:train_size])

for i in range(test_size):
    model = train_model(np.array(current_history))
    model.eval()
    with torch.no_grad():
        input_seq = torch.tensor(current_history[-seq_len:], dtype=torch.float32).view(1, 1, seq_len)
        pred = model(input_seq).numpy().flatten()
        # Multi-step ahead: we only take the first step of the 12-step forecast for the rolling evaluation
        all_forecasts.append(float(pred[0]))
    
    # Update history with ground truth (rolling update)
    current_history.append(data[train_size + i])

print(all_forecasts)