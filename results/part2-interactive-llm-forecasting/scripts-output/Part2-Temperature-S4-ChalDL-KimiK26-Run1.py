import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback to cpu if cuda is unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fixed hyperparameters
input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

# load and preprocess data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
target_col = 'Daily minimum temperatures'
df[target_col] = pd.to_numeric(df[target_col], errors='coerce')
series = df[target_col].to_numpy()

# chronological split: first 2921 observations for training
train_size = 2921
train_series = series[:train_size]
test_series = series[train_size:]

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_samples = len(data) - seq_len - pred_len + 1

    def __len__(self):
        return max(0, self.n_samples)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return torch.FloatTensor(x).unsqueeze(-1), torch.FloatTensor(y)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

forecasts = []
current_train = train_series.copy()
test_len = len(test_series)
block_size = 30
num_blocks = (test_len + block_size - 1) // block_size

for b in range(num_blocks):
    dataset = TimeSeriesDataset(current_train, seq_len, pred_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        input_seq = current_train[-seq_len:]
        x_input = torch.FloatTensor(input_seq).unsqueeze(0).unsqueeze(-1).to(device)
        pred_block = model(x_input).cpu().numpy().flatten()

    start_idx = b * block_size
    end_idx = min(start_idx + block_size, test_len)
    needed = end_idx - start_idx
    forecasts.extend(pred_block[:needed].tolist())

    true_block = test_series[start_idx:end_idx]
    current_train = np.concatenate([current_train, true_block])

print(forecasts)