import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler

# Set seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# TCN Model Components
class ChausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ChausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = ChausalConv1d(n_inputs, n_outputs, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = ChausalConv1d(n_outputs, n_outputs, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

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
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, 1, dilation_size, dropout)]
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        y1 = self.network(x.transpose(1, 2))
        return self.linear(y1[:, :, -1])

# Dataset class
class TSData(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, 1] # Target is Ice cream
        return torch.FloatTensor(x), torch.FloatTensor(y)

# Load and Preprocess
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
data_values = df[['Heater', 'Ice cream']].values.astype(float)

train_size = 158
test_size = 40
seq_len = 12
pred_len = 12
batch_size = 16
epochs = 30
lr = 0.001

# Forecasting logic
all_forecasts = []
current_train_data = data_values[:train_size]
test_data = data_values[train_size:]

# Block-wise rolling update
num_blocks = int(np.ceil(test_size / pred_len))

for b in range(num_blocks):
    # Scale data
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(current_train_data)
    
    # Prepare DataLoader
    ds = TSData(train_scaled, seq_len, pred_len)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    
    # Initialize Model
    model = TCN(input_size=2, output_size=12, num_channels=[32, 32, 32], kernel_size=3, dropout=0.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Train
    model.train()
    for epoch in range(epochs):
        for xb, yb in dl:
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            
    # Forecast
    model.eval()
    with torch.no_grad():
        last_window = train_scaled[-seq_len:].reshape(1, seq_len, 2)
        pred_scaled = model(torch.FloatTensor(last_window)).numpy().flatten()
        
        # Inverse transform (only target column)
        # Create dummy array for inverse transform
        dummy = np.zeros((pred_len, 2))
        dummy[:, 1] = pred_scaled
        pred_final = scaler.inverse_transform(dummy)[:, 1]
        
        # Determine how many steps to take from this block
        steps_to_keep = min(pred_len, test_size - len(all_forecasts))
        all_forecasts.extend(pred_final[:steps_to_keep].tolist())
        
    # Update training data with ground truth for next block
    if b < num_blocks - 1:
        actual_next_block = test_data[b*pred_len : (b+1)*pred_len]
        current_train_data = np.vstack([current_train_data, actual_next_block])

print(all_forecasts)