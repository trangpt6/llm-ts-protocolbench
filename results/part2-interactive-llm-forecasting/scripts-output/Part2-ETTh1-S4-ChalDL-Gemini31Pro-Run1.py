import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load and preprocess data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Feature engineering to match input_size of 7
df['hour'] = df.index.hour
df['dayofweek'] = df.index.dayofweek
df['day'] = df.index.day
df['month'] = df.index.month
df['dayofyear'] = df.index.dayofyear
df['week'] = df.index.isocalendar().week.astype(float)

data = df.values
target_idx = 0

# Train/test split
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Scaling
scaler = StandardScaler()
train_data_scaled = scaler.fit_transform(train_data)
test_data_scaled = scaler.transform(test_data)

# Hyperparameters
input_size = 7
seq_len = 168
pred_len = 168
hidden_size = 64
num_layers = 4
kernel_size = 4
dilations = [1, 2, 4, 8, 16, 32]
epochs = 20
batch_size = 32
lr = 0.001

# Sequence creation
def create_sequences(data_array, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data_array) - seq_len - pred_len + 1):
        X.append(data_array[i:i+seq_len])
        y.append(data_array[i+seq_len:i+seq_len+pred_len, target_idx])
    return np.array(X), np.array(y)

# TCN Model components
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
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, pred_len, hidden_size, num_layers, kernel_size, dilations):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation_size = dilations[i] if i < len(dilations) else 2**i
            in_channels = input_size if i == 0 else hidden_size
            out_channels = hidden_size
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=0.2)]
        self.tcn = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.tcn(x)
        y = y[:, :, -1]
        out = self.linear(y)
        return out

# Training function
def train_model(model, train_X, train_y, epochs, batch_size, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(train_X, dtype=torch.float32), torch.tensor(train_y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

# Initial training
train_X, train_y = create_sequences(train_data_scaled, seq_len, pred_len)
model = TCNModel(input_size, pred_len, hidden_size, num_layers, kernel_size, dilations).to(device)
train_model(model, train_X, train_y, epochs, batch_size, lr)

# Block-wise rolling update forecasting
forecasts = []
current_train_data = train_data_scaled.copy()
test_steps = len(test_data_scaled)

for i in range(0, test_steps, pred_len):
    model.eval()
    last_seq = current_train_data[-seq_len:]
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        pred = model(last_seq_tensor).cpu().numpy()[0]
    
    block_size = min(pred_len, test_steps - i)
    
    dummy = np.zeros((block_size, input_size))
    dummy[:, target_idx] = pred[:block_size]
    pred_unscaled = scaler.inverse_transform(dummy)[:, target_idx]
    
    forecasts.extend(pred_unscaled.tolist())
    
    true_block = test_data_scaled[i:i+block_size]
    current_train_data = np.vstack((current_train_data, true_block))
    
    if i + block_size < test_steps:
        new_train_X, new_train_y = create_sequences(current_train_data, seq_len, pred_len)
        model = TCNModel(input_size, pred_len, hidden_size, num_layers, kernel_size, dilations).to(device)
        train_model(model, new_train_X, new_train_y, epochs, batch_size, lr)

print(forecasts)