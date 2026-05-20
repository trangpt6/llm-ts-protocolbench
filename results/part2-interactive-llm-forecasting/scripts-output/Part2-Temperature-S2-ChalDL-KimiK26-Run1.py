import random
import numpy as np
import torch
import torch.nn as nn
import pandas as pd

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# use cpu to ensure compatibility
device = torch.device('cpu')

# load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
series = df['Daily minimum temperatures'].values.astype(np.float32)

# chronological split
train_size = 2920
train_data = series[:train_size]
test_data = series[train_size:]

# fixed hyperparameters
input_size = 1
seq_len = 7
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(data, seq_len):
    X = []
    Y = []
    for i in range(len(data) - seq_len):
        x = data[i:i + seq_len]
        y = data[i + seq_len]
        X.append(x)
        Y.append(y)
    X = np.array(X).reshape(-1, seq_len, input_size)
    Y = np.array(Y).reshape(-1, pred_len)
    return torch.from_numpy(X).float(), torch.from_numpy(Y).float()

def train(model, X, Y):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(X, Y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

history = train_data.copy()
forecasts = []

for i in range(len(test_data)):
    X, Y = make_sequences(history, seq_len)
    model = GRUModel().to(device)
    train(model, X, Y)
    model.eval()
    with torch.no_grad():
        x_input = history[-seq_len:].reshape(1, seq_len, input_size)
        x_input = torch.from_numpy(x_input).float().to(device)
        pred = model(x_input).cpu().numpy()[0, 0]
    forecasts.append(float(pred))
    history = np.append(history, test_data[i])

print(forecasts)