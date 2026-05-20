import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler
import random

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

df = pd.read_csv(r'../../../data/Temperature.csv')
df.columns = ['Date', 'Value']
df['Date'] = pd.to_datetime(df['Date'], dayfirst=False)
df = df.sort_values('Date').reset_index(drop=True)
data = df['Value'].values.astype(np.float32)

train_size = 2921
test_start = train_size
test_size = len(data) - train_size

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        last_hidden = h_n[-1]
        out = self.fc(last_hidden)
        return out

def create_sequences(y, seq_len, pred_len):
    X, Y = [], []
    for i in range(len(y) - seq_len - pred_len + 1):
        X.append(y[i:i+seq_len])
        Y.append(y[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(Y)

def train_model(train_seq_X, train_seq_Y, seq_len):
    model = LSTMModel(1, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = TensorDataset(torch.tensor(train_seq_X, dtype=torch.float32),
                            torch.tensor(train_seq_Y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x.unsqueeze(-1))
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
    model.eval()
    return model

current_train = data[:train_size].copy()
forecasts = []

n_blocks = int(np.ceil(test_size / pred_len))

for i in range(n_blocks):
    scaler = MinMaxScaler()
    scaled_train = scaler.fit_transform(current_train.reshape(-1, 1)).flatten()
    X_seq, Y_seq = create_sequences(scaled_train, seq_len, pred_len)
    model = train_model(X_seq, Y_seq, seq_len)

    last_input = scaled_train[-seq_len:]
    model_input = torch.tensor(last_input.reshape(1, seq_len, 1), dtype=torch.float32).to(device)
    with torch.no_grad():
        scaled_pred = model(model_input).cpu().numpy().flatten()
    pred = scaler.inverse_transform(scaled_pred.reshape(-1, 1)).flatten()

    block_start = i * pred_len
    block_end = min((i+1) * pred_len, test_size)
    take = block_end - block_start
    forecasts.extend(pred[:take].tolist())

    true_block = data[test_start + block_start : test_start + block_end]
    current_train = np.concatenate([current_train, true_block])

print(forecasts)