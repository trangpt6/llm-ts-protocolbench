import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Set seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Read dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df.columns = df.columns.str.strip()
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

target_col = 'Ice cream'
exog_col = 'Heater'

# Chronological split
train_size = int(0.8 * len(df))  # 158
test_size = len(df) - train_size  # 40
train = df.iloc[:train_size].copy()
test = df.iloc[train_size:].copy()

# Build sliding windows for training
seq_len = 12
pred_len = 1
input_size = 2  # Heater and Ice cream

X_list, y_list = [], []
# Only use training set for constructing windows, but target is next step within training
for i in range(seq_len, len(train)):
    window = train.iloc[i-seq_len:i][[exog_col, target_col]].values  # shape (seq_len, 2)
    target_val = train.iloc[i][target_col]
    X_list.append(window)
    y_list.append(target_val)

X_train = np.array(X_list)  # (samples, seq_len, 2)
y_train = np.array(y_list)  # (samples,)

# Convert to torch tensors
# TCN expects input (batch, channels, seq_len)
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)  # (batch, 2, seq_len)
y_train_t = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

dataset = TensorDataset(X_train_t, y_train_t)
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Define TCN module
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.conv.weight, 0, 0.01)
        nn.init.constant_(self.conv.bias, 0)

    def forward(self, x):
        out = self.conv(x)
        # causal: remove future padding from the right end
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        return self.dropout(out)

class ResidualTCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super().__init__()
        self.causal1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation, dropout)
        self.causal2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation, dropout)
        self.relu = nn.ReLU()
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.init_weights()

    def init_weights(self):
        if isinstance(self.residual, nn.Conv1d):
            nn.init.normal_(self.residual.weight, 0, 0.01)
            nn.init.constant_(self.residual.bias, 0)

    def forward(self, x):
        residual = self.residual(x)
        out = self.relu(self.causal1(x))
        out = self.relu(self.causal2(out))
        return self.relu(out + residual)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super().__init__()
        layers = []
        in_ch = input_size
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else dilations[-1]
            out_ch = hidden_size if i < num_layers-1 else hidden_size  # keep same across blocks
            layers.append(ResidualTCNBlock(in_ch, out_ch, kernel_size, dilation))
            in_ch = out_ch
        self.network = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_size, pred_len)
        self.init_weights()

    def init_weights(self):
        nn.init.normal_(self.output.weight, 0, 0.01)
        nn.init.constant_(self.output.bias, 0)

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.network(x)
        # take last time step's hidden representation
        out = out[:, :, -1]
        return self.output(out)

# Hyperparameters
hidden_size = 32
num_layers = 2
kernel_size = 3
dilations = [1, 2, 4, 8]  # will use only first num_layers

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len=1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * X_batch.size(0)
    # no print required

model.eval()

# Recursive forecasting for test horizon
# Start context: last seq_len points from training data
context = train.iloc[-seq_len:][[exog_col, target_col]].copy()  # (seq_len, 2) with true values
forecasts = []

# Prepare test exogenous (Heater) as a numpy array for indexing
test_exog = test[exog_col].values  # length test_size

for i in range(test_size):
    # build input window: current context (seq_len, 2)
    win = context.values  # shape (seq_len, 2)
    # convert to tensor (1, 2, seq_len)
    win_t = torch.tensor(win, dtype=torch.float32).unsqueeze(0).permute(0, 2, 1).to(device)
    with torch.no_grad():
        next_pred = model(win_t).cpu().item()
    forecasts.append(next_pred)
    # update context: drop oldest, append new point with predicted Ice cream and known Heater
    next_heater = test_exog[i]
    new_row = pd.DataFrame([[next_heater, next_pred]], columns=[exog_col, target_col])
    context = pd.concat([context.iloc[1:], new_row], ignore_index=True)

print(forecasts)