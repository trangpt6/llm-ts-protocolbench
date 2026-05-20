import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
data = df['Daily minimum temperatures'].values.astype(float)
train_data = data[:2920]
test_data = data[2920:]
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
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
        out = self.network(x)
        out = out[:, :, -1]
        out = self.fc(out)
        return out
input_size = 1
seq_len = 14
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8]
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
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()
for epoch in range(epochs):
    for batch_x, batch_y in loader:
        optimizer.zero_grad()
        output = model(batch_x)
        loss = criterion(output, batch_y)
        loss.backward()
        optimizer.step()
forecasts = []
current_window = train_data[-seq_len:].tolist()
for _ in range(len(test_data)):
    input_seq = torch.tensor(current_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(input_seq).item()
    forecasts.append(pred)
    current_window = current_window[1:] + [pred]
print(forecasts)