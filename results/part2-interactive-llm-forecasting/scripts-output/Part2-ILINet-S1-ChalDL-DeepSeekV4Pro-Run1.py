import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader, TensorDataset

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Read CSV
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df = df.set_index('DATE')

# Resample to weekly (Sunday) to fill missing dates like 2002-01-06
df = df.resample('W-SUN').asfreq()
# Forward fill target only for the missing date, keep zeros where present
target_col = '% WEIGHTED ILI'
df[target_col] = df[target_col].fillna(method='ffill')

# Chronological split by dates
train_end = '2017-11-26'
test_start = '2017-12-03'
train = df.loc[:train_end, target_col].values.astype(np.float32)
test = df.loc[test_start:, target_col].values.astype(np.float32)

# Scale using MinMaxScaler on train set
min_val = train.min()
max_val = train.max()
train_scaled = (train - min_val) / (max_val - min_val + 1e-8)
test_scaled = (test - min_val) / (max_val - min_val + 1e-8)

# Hyperparameters
seq_len = 52
pred_len = 1
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations_orig = [1, 2, 4, 8, 16, 32]
dilations = dilations_orig[:num_layers]  # Use only as many dilations as layers
epochs = 50
batch_size = 32
lr = 0.001

# Create sequences for training
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, seq_len, pred_len)
X_train = torch.tensor(X_train).unsqueeze(-1)  # (samples, seq_len, input_size)
y_train = torch.tensor(y_train).squeeze(-1)    # (samples, pred_len=1)

# Define TCN model
class CausalConv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(CausalConv1dBlock, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=self.padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        out = self.conv(x)
        # Remove the last padding elements to maintain causality
        out = out[:, :, :-self.padding]
        out = self.relu(out)
        out = self.dropout(out)
        return out

class TCN(nn.Module):
    def __init__(self, input_size, seq_len, hidden_size, num_layers, kernel_size, dilations, output_size=pred_len):
        super(TCN, self).__init__()
        layers = []
        in_channels = input_size
        for i in range(num_layers):
            out_channels = hidden_size if i < num_layers - 1 else output_size
            dilation = dilations[i]
            layers.append(CausalConv1dBlock(in_channels, hidden_size, kernel_size, dilation))
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_size, output_size)
        # Actually the last block should output output_size? We'll adjust: separate final conv
        # Redesign: sequence of hidden conv blocks, then a final 1x1 conv to output
        self.conv_blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation = dilations[i]
            self.conv_blocks.append(
                CausalConv1dBlock(input_size if i == 0 else hidden_size, hidden_size, kernel_size, dilation)
            )
        self.final_conv = nn.Conv1d(hidden_size, output_size, 1)
        self.seq_len = seq_len

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.transpose(1, 2)  # (batch, input_size, seq_len)
        for conv in self.conv_blocks:
            x = conv(x)
        out = self.final_conv(x)  # (batch, output_size, seq_len)
        # Take the last time step prediction
        out = out[:, :, -1]  # (batch, output_size)
        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(input_size=1, seq_len=seq_len, hidden_size=hidden_size, num_layers=num_layers,
            kernel_size=kernel_size, dilations=dilations).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training
model.train()
for epoch in range(epochs):
    total_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    if (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.6f}')

# Recursive forecasting on test set
model.eval()
forecasts_scaled = []
# Last sequence from training set
current_seq = train_scaled[-seq_len:].copy()
for _ in range(len(test)):
    input_t = torch.tensor(current_seq[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(input_t).cpu().item()
    forecasts_scaled.append(pred)
    # Append prediction and slide window
    current_seq = np.append(current_seq, pred)

forecasts = [p * (max_val - min_val) + min_val for p in forecasts_scaled]
print(forecasts)