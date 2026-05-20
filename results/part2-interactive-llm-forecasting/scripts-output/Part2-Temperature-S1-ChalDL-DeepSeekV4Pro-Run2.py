import random
import numpy as np
import torch
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

import pandas as pd
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

# Read dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date').reset_index(drop=True)
series = df['Daily minimum temperatures'].values.astype(np.float32)

# Chronological split: first 80% train
n_train = int(0.8 * len(series))  # 2921
train_series = series[:n_train]
test_series = series[n_train:]

# Hyperparameters from Turn 2
seq_len = 14
pred_len = 1
input_size = 1
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
epochs = 50
batch_size = 32
lr = 0.001

# Create sequences for training (sliding window)
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len].reshape(seq_len, 1))
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_series, seq_len, pred_len)
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# TCN implementation (simple causal convolutions with dilation)
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        in_channels = input_size
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 1
            padding = (kernel_size - 1) * dilation
            layers.append(nn.Conv1d(in_channels, hidden_size, kernel_size, padding=padding, dilation=dilation))
            layers.append(nn.ReLU())
            in_channels = hidden_size
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch, seq_len, input_size) -> transpose to (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # out shape: (batch, hidden_size, seq_len) - take last time step
        out = out[:, :, -1]
        out = self.fc(out)
        return out.squeeze(-1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training
model.train()
for epoch in range(epochs):
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    # Optional: print(f'Epoch {epoch+1}/{epochs} Loss: {epoch_loss/len(train_loader):.4f}')

# Recursive forecasting on test set
model.eval()
forecasts = []
# Initial input sequence is last seq_len points from training set
current_seq = torch.tensor(train_series[-seq_len:].reshape(1, seq_len, 1), dtype=torch.float32).to(device)

with torch.no_grad():
    for _ in range(len(test_series)):
        # Predict next value
        pred = model(current_seq).cpu().item()
        forecasts.append(pred)
        # Shift window: drop first element, append prediction
        current_seq = current_seq.squeeze(0).tolist()
        current_seq.append([pred])
        current_seq = current_seq[1:]
        current_seq = torch.tensor([current_seq], dtype=torch.float32).to(device)

# Ensure exactly len(test_series) forecasts
forecasts = forecasts[:len(test_series)]

print(forecasts)