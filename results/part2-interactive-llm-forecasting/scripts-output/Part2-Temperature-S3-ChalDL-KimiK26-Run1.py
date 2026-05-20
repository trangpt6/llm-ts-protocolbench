import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

df = pd.read_csv(r'../../../data/Temperature.csv')
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
df = df.sort_values('Date')
values = df['Daily minimum temperatures'].values.astype(np.float32)

train_size = 2920
train_values = values[:train_size]
test_values = values[train_size:]

class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    X = np.array(X).reshape(-1, seq_len, 1)
    y = np.array(y)
    return X, y

seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

history = train_values.copy()
forecasts = []

for i in range(len(test_values)):
    X, y = create_sequences(history, seq_len, pred_len)
    model = GRUModel(1, hidden_size, num_layers, pred_len)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        dataset = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    x_input = history[-seq_len:].reshape(1, seq_len, 1)
    with torch.no_grad():
        pred = model(torch.from_numpy(x_input))
    forecasts.append(float(pred[0, 0]))
    history = np.append(history, test_values[i])

print(forecasts)