import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Load dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']
data = df[feature_cols].values.astype(np.float32)

# Train/Test split
total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size
train_data = data[:train_size]
test_data = data[train_size:]

# Hyperparameters
input_size = 2
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 30
batch_size = 16
lr = 0.001

class ChausalConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(ChausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding, dilation=dilation)

    def forward(self, x):
        x = self.conv(x)
        return x[:, :, :-self.padding]

class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super(TCNBlock, self).__init__()
        self.conv1 = ChausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = ChausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.relu2 = nn.ReLU()
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super(TCNModel, self).__init__()
        layers = []
        for i in range(num_layers):
            in_c = input_size if i == 0 else hidden_size
            layers.append(TCNBlock(in_c, hidden_size, kernel_size, dilations[i % len(dilations)]))
        self.tcn = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.transpose(1, 2)
        out = self.tcn(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out

class TSDataSet(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len

    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len, 1]
        return x, y

def train_model(model, train_data_subset):
    dataset = TSDataSet(train_data_subset, seq_len, pred_len)
    if len(dataset) == 0: return model
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
    return model

all_forecasts = []
current_train_data = train_data.copy()
test_idx = 0
block_size = 12

while test_idx < test_size:
    model = TCNModel(input_size, hidden_size, num_layers, kernel_size, dilations, pred_len)
    model = train_model(model, current_train_data)
    
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(current_train_data[-seq_len:]).unsqueeze(0)
        pred = model(last_seq).squeeze(0).numpy()
    
    steps_to_add = min(block_size, test_size - test_idx)
    all_forecasts.extend(pred[:steps_to_add].tolist())
    
    # Update training data with ground truth for next block
    new_block = test_data[test_idx : test_idx + steps_to_add]
    current_train_data = np.concatenate([current_train_data, new_block], axis=0)
    test_idx += steps_to_add

print(all_forecasts)