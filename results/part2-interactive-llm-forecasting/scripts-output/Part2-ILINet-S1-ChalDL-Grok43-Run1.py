import pandas as pd
import numpy as np
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target_col = '% WEIGHTED ILI'
for col in df.columns:
    if col != 'DATE' and col != target_col:
        df[col] = df[col].ffill()
series = df[target_col].values.astype(float)
train_size = 1019
train_data = series[:train_size]
test_data = series[train_size:]
class TemporalBlock(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, dilation):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = torch.nn.Conv1d(n_inputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.conv2 = torch.nn.Conv1d(n_outputs, n_outputs, kernel_size, padding=padding, dilation=dilation)
        self.downsample = torch.nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)
class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super().__init__()
        layers = []
        for i in range(num_layers):
            dilation = dilations[i] if i < len(dilations) else 1
            layers.append(TemporalBlock(input_size if i == 0 else hidden_size, hidden_size, kernel_size, dilation))
        self.network = torch.nn.Sequential(*layers)
        self.linear = torch.nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        y = self.network(x)
        y = y[:, :, -1]
        y = self.linear(y)
        return y
input_size = 1
seq_len = 52
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32]
epochs = 50
batch_size = 32
lr = 0.001
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_data, seq_len)
X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = torch.nn.MSELoss()
model.train()
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size].to(device)
        batch_y = y_train[i:i+batch_size].to(device)
        optimizer.zero_grad()
        output = model(batch_X)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
model.eval()
forecasts = []
current_seq = train_data[-seq_len:].tolist()
for _ in range(len(test_data)):
    input_seq = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    with torch.no_grad():
        pred = model(input_seq).item()
    forecasts.append(pred)
    current_seq = current_seq[1:] + [pred]
print(forecasts)