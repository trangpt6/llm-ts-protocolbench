import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# load and sort data
df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
values = df['Daily minimum temperatures'].astype(float).values

# fixed chronological split from Turn 0
train_size = 2921
test_size = 731
train_data = values[:train_size]
test_data = values[train_size:train_size + test_size]

# fixed hyperparameters
input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001

# fallback to cpu if cuda unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def create_sequences(data, seq_len, pred_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:i + seq_len]
        y = data[i + seq_len:i + seq_len + pred_len]
        xs.append(x)
        ys.append(y)
    X = np.array(xs).reshape(-1, seq_len, input_size)
    y = np.array(ys)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(model, data):
    model.train()
    X, y = create_sequences(data, seq_len, pred_len)
    if len(X) == 0:
        return
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

# initial training on train set
model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
train_model(model, train_data)

history = train_data.copy()
predictions = []
num_blocks = (test_size + pred_len - 1) // pred_len

for b in range(num_blocks):
    model.eval()
    with torch.no_grad():
        x_input = history[-seq_len:].reshape(1, seq_len, input_size)
        x_tensor = torch.tensor(x_input, dtype=torch.float32).to(device)
        pred = model(x_tensor).cpu().numpy().flatten()

    remaining = test_size - len(predictions)
    block_len = min(pred_len, remaining)
    predictions.extend(pred[:block_len].tolist())

    true_block = test_data[len(predictions) - block_len:len(predictions)]
    history = np.concatenate([history, true_block])

    if len(predictions) < test_size:
        model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
        train_model(model, history)

forecasts = predictions[:test_size]
print(forecasts)