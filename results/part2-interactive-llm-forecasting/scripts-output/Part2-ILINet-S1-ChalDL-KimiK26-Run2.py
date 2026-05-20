import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

# Reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device configuration
device = torch.device('cpu')

# Load and sort data
df = pd.read_csv(r'../../../data/ILINet.csv')
df['DATE'] = pd.to_datetime(df['DATE'])
df = df.sort_values('DATE').reset_index(drop=True)

# Preprocess: convert empty strings to NaN in specified age columns
for col in ['AGE 25-49', 'AGE 50-64']:
    df[col] = pd.to_numeric(df[col].replace(r'^\s*$', np.nan, regex=True), errors='coerce')

# Preprocess: mask target zeros from suspended reporting periods (1998-2002 summers)
mask_missing = (df['NUM. OF PROVIDERS'] == 0) & (df['TOTAL PATIENTS'] == 0) & (df['DATE'].dt.year <= 2002)
df.loc[mask_missing, '% WEIGHTED ILI'] = np.nan

# Extract target and apply chronological split
target = df['% WEIGHTED ILI'].values.astype(float)
train_size = 1044
test_size = 262
train_target = target[:train_size]

# Scale based on non-NaN training data only
scaler = StandardScaler()
valid_train_mask = ~np.isnan(train_target)
scaler.fit(train_target[valid_train_mask].reshape(-1, 1))
scaled = np.empty_like(target, dtype=float)
scaled[:] = np.nan
valid_mask = ~np.isnan(target)
scaled[valid_mask] = scaler.transform(target[valid_mask].reshape(-1, 1)).flatten()
train_scaled = scaled[:train_size]

# PyTorch Dataset that skips windows containing any NaN
class ILIDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.indices = []
        limit = len(data) - seq_len - pred_len + 1
        for i in range(limit):
            if not np.isnan(data[i:i + seq_len + pred_len]).any():
                self.indices.append(i)
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, idx):
        i = self.indices[idx]
        x = self.data[i:i + self.seq_len]
        y = self.data[i + self.seq_len:i + self.seq_len + self.pred_len]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y).unsqueeze(-1)

seq_len = 52
pred_len = 1
dataset = ILIDataset(train_scaled, seq_len, pred_len)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# TCN components
class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.0):
        super(TemporalBlock, self).__init__()
        self.pad = nn.ConstantPad1d(((kernel_size - 1) * dilation, 0), 0)
        self.conv1 = nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.final_relu = nn.ReLU()
    def forward(self, x):
        out = self.conv1(self.pad(x))
        out = self.relu(out)
        out = self.dropout(out)
        return self.final_relu(out + self.downsample(x))

class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dilations, dropout=0.0):
        super(TCN, self).__init__()
        layers = []
        for i, d in enumerate(dilations):
            in_ch = input_size if i == 0 else num_channels
            layers.append(TemporalBlock(in_ch, num_channels, kernel_size, d, dropout))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels, output_size)
    def forward(self, x):
        x = x.transpose(1, 2)
        y = self.network(x)
        y = y[:, :, -1]
        return self.linear(y)

model = TCN(input_size=1, output_size=1, num_channels=64, kernel_size=3, dilations=[1, 2, 4, 8, 16, 32]).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Training loop
for epoch in range(50):
    model.train()
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()

# Recursive forecasting over test set without using ground truth
model.eval()
history = [float(v) for v in train_scaled[-seq_len:]]
assert len(history) == seq_len and not any(np.isnan(history)), 'Invalid initial history'
forecasts_scaled = []
with torch.no_grad():
    for _ in range(test_size):
        x_input = torch.FloatTensor(history[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(x_input).cpu().item()
        forecasts_scaled.append(pred)
        history.append(pred)

forecasts = scaler.inverse_transform(np.array(forecasts_scaled).reshape(-1, 1)).flatten().tolist()
print(forecasts)