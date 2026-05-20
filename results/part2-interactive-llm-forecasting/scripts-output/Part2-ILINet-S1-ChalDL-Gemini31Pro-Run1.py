import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load data
df = pd.read_csv(r'../../../data/ILINet.csv')

# Preprocessing
df['DATE'] = pd.to_datetime(df['DATE'])
df.set_index('DATE', inplace=True)
df = df.asfreq('W-SUN')
df['% WEIGHTED ILI'] = df['% WEIGHTED ILI'].interpolate(method='linear')

# Extract target
data = df['% WEIGHTED ILI'].values.reshape(-1, 1)

# Train/test split
total_timesteps = len(data)
train_size = int(0.8 * total_timesteps)
train_data = data[:train_size]

# Scale data
scaler = StandardScaler()
train_scaled = scaler.fit_transform(train_data)

# Hyperparameters
input_size = 1
seq_len = 52
pred_len = 1
hidden_size = 64
num_layers = 3
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

X_train = torch.tensor(np.array(X_train), dtype=torch.float32)
y_train = torch.tensor(np.array(y_train), dtype=torch.float32)

dataset = TensorDataset(X_train, y_train)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define TCN Model
class CausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)
        
    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super().__init__()
        layers = []
        in_channels = input_size
        for _ in range(num_layers):
            for d in dilations:
                layers.append(CausalConv1d(in_channels, hidden_size, kernel_size, d))
                layers.append(nn.ReLU())
                in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

# Training
model.train()
for epoch in range(epochs):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()

# Recursive Forecasting
model.eval()
forecasts_scaled = []
current_seq = train_scaled[-seq_len:].copy()
test_len = total_timesteps - train_size

with torch.no_grad():
    for _ in range(test_len):
        x_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(x_tensor).cpu().numpy()[0, 0]
        forecasts_scaled.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

# Inverse transform
forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten().tolist()

# Print final output
print(forecasts)