import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
df.sort_values('DATE', inplace=True)

# Split by date exactly as defined
train_df = df[df['DATE'] <= '2017-12-10'].copy()
test_df = df[df['DATE'] > '2017-12-10'].copy()

# Preprocess training data: remove rows where all numeric columns are zero
# First, fill missing numeric values with 0 temporarily to detect summer shutdown rows
num_cols = [col for col in df.columns if col != 'DATE']
for col in num_cols:
    train_df[col] = pd.to_numeric(train_df[col], errors='coerce').fillna(0)
all_zero_mask = (train_df[num_cols] == 0).all(axis=1)
train_clean = train_df[~all_zero_mask].copy()

# Extract target series (one-dimensional)
train_series = train_clean['% WEIGHTED ILI'].values.astype(np.float32)
test_series = test_df['% WEIGHTED ILI'].values.astype(np.float32)

seq_len = 52
pred_len = 1
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4]  # use first num_layers dilations to match num_layers=3
epochs = 50
batch_size = 32
lr = 0.001

# Build sliding windows from training series
X_train = []
y_train = []
for i in range(len(train_series) - seq_len):
    X_train.append(train_series[i:i+seq_len])
    y_train.append(train_series[i+seq_len])
X_train = np.array(X_train).reshape(-1, 1, seq_len)  # (samples, channels=1, seq_len)
y_train = np.array(y_train).reshape(-1, pred_len)

train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define TCN model
class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=0, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    def forward(self, x):
        out = F.pad(x, (self.padding, 0))
        out = self.conv(out)
        out = out[:, :, :x.size(2)]
        out = self.relu(out)
        if self.dropout:
            out = self.dropout(out)
        res = self.residual(x)
        return out + res

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        in_ch = input_size
        for i, d in enumerate(dilations):
            out_ch = hidden_size if i == 0 else hidden_size
            layers.append(CausalConvBlock(in_ch, out_ch, kernel_size, d))
            in_ch = out_ch
        self.conv_layers = nn.Sequential(*layers)
        self.output_conv = nn.Conv1d(hidden_size, pred_len, 1)
    def forward(self, x):
        out = self.conv_layers(x)
        out = self.output_conv(out)
        return out[:, :, -1].squeeze(-1)  # (batch, pred_len) -> (batch, pred_len)

device = torch.device('cpu')
model = TCN(input_size=1, hidden_size=hidden_size, kernel_size=kernel_size, dilations=dilations, pred_len=pred_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
model.train()
for epoch in range(epochs):
    total_loss = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        out = model(xb)
        loss = criterion(out, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    # print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_dataset):.6f}')  # optional

# Recursive forecasting over the entire test set
model.eval()
forecasts = []
current_seq = train_series[-seq_len:].copy()  # last 52 values from training set

for step in range(len(test_series)):
    with torch.no_grad():
        inp = torch.tensor(current_seq[-seq_len:].reshape(1, 1, seq_len)).float().to(device)
        pred = model(inp).cpu().item()
    forecasts.append(pred)
    # Update sequence: drop first element, append prediction
    current_seq = np.append(current_seq, pred)

print(forecasts)