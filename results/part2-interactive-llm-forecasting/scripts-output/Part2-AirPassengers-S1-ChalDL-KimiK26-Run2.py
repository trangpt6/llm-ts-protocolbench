import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# read dataset
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)
train_size = 115

# split data chronologically
train_data = data[:train_size]

# create sliding windows for training
seq_len = 12
X_list = []
y_list = []
for i in range(seq_len, len(train_data)):
    X_list.append(train_data[i - seq_len:i])
    y_list.append(train_data[i])
X = np.array(X_list).reshape(-1, seq_len, 1)
y = np.array(y_list).reshape(-1, 1)

# prepare data loader
dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# define LSTM model
class LSTMModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = torch.nn.Linear(64, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# use cpu for compatibility
device = torch.device('cpu')
model = LSTMModel().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# train model
model.train()
for epoch in range(50):
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        optimizer.zero_grad()
        output = model(xb)
        loss = criterion(output, yb)
        loss.backward()
        optimizer.step()

# recursive one-step ahead forecasting over test set
model.eval()
history = train_data[-seq_len:].tolist()
forecasts = []
with torch.no_grad():
    for _ in range(29):
        x_input = torch.FloatTensor(history[-seq_len:]).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(x_input)
        val = pred.item()
        forecasts.append(val)
        history.append(val)

print(forecasts)