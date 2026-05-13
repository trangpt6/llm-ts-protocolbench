import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# set seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

df = pd.read_csv(r'../../../data/AirPassengers.csv', header=0)
df = df.sort_values(by=df.columns[0])
series = df.iloc[:, 1].astype(np.float32).to_numpy()

train_size = 115
train = series[:train_size]
test = series[train_size:]

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

def make_sequences(data, seq_len):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
forecasts = []
current_train = train.copy()

for step in range(len(test)):
    x_train, y_train = make_sequences(current_train, seq_len)
    x_train = torch.from_numpy(x_train).view(-1, seq_len, input_size).to(device)
    y_train = torch.from_numpy(y_train).view(-1, pred_len).to(device)
    
    model = GRUModel().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(x_train.size(0))
        for i in range(0, x_train.size(0), batch_size):
            idx = perm[i:i+batch_size]
            batch_x = x_train[idx]
            batch_y = y_train[idx]
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    with torch.no_grad():
        x_input = torch.from_numpy(current_train[-seq_len:]).view(1, seq_len, input_size).to(device)
        pred = model(x_input).item()
    forecasts.append(pred)
    current_train = np.append(current_train, test[step])

print(forecasts)