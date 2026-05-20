import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values.astype(np.float32)
target = target.reshape(-1, 1)  # univariate

total_len = len(target)
train_size = int(0.8 * total_len)
test_size = total_len - train_size

train_data = target[:train_size].copy()
test_data = target[train_size:].copy()
test_len = test_size

seq_len = 52
pred_len = 52
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 32
lr = 0.001

class LSTMForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, pred_len=52):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # last hidden state
        out = self.fc(out)
        return out

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len].flatten())
    return np.array(X), np.array(y)

forecasts = []
current_train = train_data.copy()

for block_start in range(0, test_len, pred_len):
    block_end = min(block_start + pred_len, test_len)
    current_pred_len = block_end - block_start

    # rebuild model each block (retrain from scratch)
    model = LSTMForecaster(input_size=1, hidden_size=hidden_size,
                           num_layers=num_layers, pred_len=pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # create sequences from current_train
    X_train, y_train = create_sequences(current_train, seq_len, pred_len)
    X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            out = model(batch_X)
            loss = criterion(out, batch_y)
            loss.backward()
            optimizer.step()

    # forecast next block
    model.eval()
    with torch.no_grad():
        last_seq = torch.tensor(current_train[-seq_len:], dtype=torch.float32).unsqueeze(0).to(device)
        pred = model(last_seq).cpu().numpy().flatten()
        if current_pred_len < pred_len:
            pred = pred[:current_pred_len]
        forecasts.extend(pred.tolist())

    # append true values of this block to training data
    true_block = test_data[block_start:block_end]
    current_train = np.concatenate([current_train, true_block])

print(forecasts)