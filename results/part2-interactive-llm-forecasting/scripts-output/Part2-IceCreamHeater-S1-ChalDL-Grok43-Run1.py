import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)
train_data = data[:158]
class TCN(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations):
        super(TCN, self).__init__()
        self.layers = torch.nn.ModuleList()
        for i in range(num_layers):
            dilation = dilations[i]
            in_channels = input_size if i == 0 else hidden_size
            self.layers.append(torch.nn.Conv1d(in_channels, hidden_size, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation))
            self.layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Linear(hidden_size, 1)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            if isinstance(layer, torch.nn.Conv1d):
                x = layer(x)
            else:
                x = layer(x)
        x = x[:, :, -1]
        x = self.fc(x)
        return x.squeeze(-1)
def create_sequences(data, seq_len):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len, 1])
    return np.array(X), np.array(y)
seq_len = 12
X_train, y_train = create_sequences(train_data, seq_len)
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
batch_size = 16
dataset = TensorDataset(X_train, y_train)
loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
model = TCN(2, 32, 2, 3, [1, 2, 4, 8])
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
epochs = 50
for epoch in range(epochs):
    model.train()
    for X_batch, y_batch in loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
current_seq = train_data[-seq_len:].copy()
forecasts = []
last_heater = train_data[-1, 0]
for _ in range(40):
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor(current_seq, dtype=torch.float32).unsqueeze(0)
        pred = model(input_tensor).item()
    forecasts.append(pred)
    current_seq = np.roll(current_seq, -1, axis=0)
    current_seq[-1, 0] = last_heater
    current_seq[-1, 1] = pred
print(forecasts)