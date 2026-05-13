import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
y = df['Passengers'].values.astype(np.float32)
n_total = len(y)
train_size = int(0.8 * n_total)
train_data = y[:train_size]
test_data = y[train_size:]

input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 32
num_layers = 1
epochs = 10
batch_size = 8
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

def create_sequences(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x = data[i:i+seq_len]
        y_val = data[i+seq_len]
        xs.append(x)
        ys.append(y_val)
    xs = np.array(xs)
    ys = np.array(ys)
    return xs, ys

def train_model(model, data):
    model.train()
    x_seq, y_seq = create_sequences(data, seq_len)
    if len(x_seq) == 0:
        return
    x_tensor = torch.from_numpy(x_seq).unsqueeze(-1).float().to(device)
    y_tensor = torch.from_numpy(y_seq).unsqueeze(-1).float().to(device)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

history = train_data.copy()
model = GRUModel().to(device)
train_model(model, history)

forecasts = []
for i in range(len(test_data)):
    model.eval()
    x_input = history[-seq_len:]
    x_tensor = torch.from_numpy(x_input).unsqueeze(0).unsqueeze(-1).float().to(device)
    with torch.no_grad():
        pred = model(x_tensor)
    pred_val = pred.cpu().numpy().flatten()[0]
    forecasts.append(float(pred_val))
    history = np.append(history, test_data[i])
    model = GRUModel().to(device)
    train_model(model, history)

print(forecasts)