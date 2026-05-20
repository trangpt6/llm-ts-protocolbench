import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
data = df.iloc[:,1].values.astype(np.float32)
train_size = 2920
train = data[:train_size]
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
seq_len = 14
X_train, y_train = create_sequences(train, seq_len)
X_train = X_train.reshape(-1, seq_len, 1)
y_train = y_train.reshape(-1, 1)
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=(kernel_size-1)*dilation, dilation=dilation)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 2**i
            layers.append(TemporalBlock(input_size if i==0 else hidden_size, hidden_size, kernel_size, dilation))
        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        out = self.network(x)
        out = out[:, :, -1]
        out = self.linear(out)
        return out
model = TCN(1, 64, 3, 3, [1,2,4,8])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
train_dataset = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
for epoch in range(50):
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
forecasts = []
current_seq = train[-14:].reshape(1, 14, 1)
for _ in range(730):
    with torch.no_grad():
        pred = model(torch.tensor(current_seq, dtype=torch.float32))
        forecasts.append(float(pred.item()))
        current_seq = np.concatenate([current_seq[0,1:,:], pred.reshape(1,1,1)], axis=0).reshape(1,14,1)
print(forecasts)