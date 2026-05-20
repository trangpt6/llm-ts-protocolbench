import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/ILINet.csv')
target = df['% WEIGHTED ILI'].values.astype(np.float32)
train_size = 1018
train_data = target[:train_size]
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilations):
        super(TCN, self).__init__()
        layers = []
        for i, d in enumerate(dilations):
            in_ch = input_size if i == 0 else hidden_size
            layers.append(nn.Conv1d(in_ch, hidden_size, kernel_size, dilation=d, padding=(kernel_size-1)*d))
            layers.append(nn.ReLU())
        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = x[:, :, -1]
        x = self.fc(x)
        return x
input_size = 1
seq_len = 52
hidden_size = 64
kernel_size = 3
dilations = [1, 2, 4, 8, 16, 32]
epochs = 50
batch_size = 32
lr = 0.001
model = TCN(input_size, hidden_size, kernel_size, dilations)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
X_train, y_train = create_sequences(train_data, seq_len)
X_train = torch.tensor(X_train).float().unsqueeze(-1)
y_train = torch.tensor(y_train).float().unsqueeze(-1)
model.train()
for epoch in range(epochs):
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i+batch_size]
        batch_y = y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
model.eval()
forecasts = []
current_window = train_data[-seq_len:].tolist()
for _ in range(255):
    input_tensor = torch.tensor(current_window).float().unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(input_tensor).item()
    forecasts.append(pred)
    current_window = current_window[1:] + [pred]
print(forecasts)