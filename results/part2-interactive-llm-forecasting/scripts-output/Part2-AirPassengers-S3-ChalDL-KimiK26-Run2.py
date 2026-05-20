import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback to cpu
device = torch.device('cpu')

# read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# no preprocessing needed per Turn 1
target_col = 'Passengers'
values = df[target_col].values.astype(np.float32)

# train/test split: first 115 for train, rest 29 for test
train_size = 115
train_values = values[:train_size]
test_values = values[train_size:]
test_len = len(test_values)

# fixed hyperparameters
input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 32
num_layers = 2
epochs = 10
batch_size = 8
lr = 0.01
kernel_size = 3
dilations = [1, 2, 4, 8]

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, padding):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation)
        self.relu2 = nn.ReLU()
        self.net = nn.Sequential(self.conv1, self.relu1, self.conv2, self.relu2)
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        in_channels = input_size
        for i in range(num_layers):
            dilation = dilations[i % len(dilations)]
            padding = (kernel_size - 1) * dilation
            out_channels = hidden_size
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation, padding))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, pred_len)
    
    def forward(self, x):
        # x: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # (batch, input_size, seq_len)
        out = self.network(x)  # (batch, hidden_size, seq_len)
        out = out[:, :, -1]  # (batch, hidden_size)
        out = self.fc(out)  # (batch, pred_len)
        return out

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)

def train_model(model, X_train, y_train, epochs, batch_size, lr):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    dataset = torch.utils.data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

# rolling forecast with retraining at each step
forecasts = []
history = list(train_values)

for t in range(test_len):
    # prepare training data from history
    hist_arr = np.array(history, dtype=np.float32)
    X, y = create_sequences(hist_arr, seq_len, pred_len)
    if len(X) == 0:
        # not enough data, use last known value
        forecasts.append(float(history[-1]))
        history.append(float(test_values[t]))
        continue
    
    X = X.reshape(-1, seq_len, input_size)
    y = y.reshape(-1, pred_len)
    
    # initialize and train model
    model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations).to(device)
    train_model(model, X, y, epochs, batch_size, lr)
    
    # predict: use last seq_len values from history
    model.eval()
    with torch.no_grad():
        last_seq = np.array(history[-seq_len:], dtype=np.float32).reshape(1, seq_len, input_size)
        last_seq_tensor = torch.from_numpy(last_seq).to(device)
        pred = model(last_seq_tensor).cpu().numpy()[0]  # (pred_len,)
    
    # append the first prediction for this step
    forecasts.append(float(pred[0]))
    
    # update history with true value (ground truth enabled)
    history.append(float(test_values[t]))

print(forecasts)