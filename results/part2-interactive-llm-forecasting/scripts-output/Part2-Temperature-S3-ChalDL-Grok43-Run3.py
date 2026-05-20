import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
# Device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data = pd.read_csv(r'../../../data/Temperature.csv')
target = data['Daily minimum temperatures'].astype(float).values
for i in range(7, len(target)):
    if target[i] == 0:
        target[i] = np.median(target[i-7:i])
train_size = 2920
train_data = target[:train_size].tolist()
test_data = target[train_size:].tolist()
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
forecasts = []
current_data = train_data[:]
for step in range(len(test_data)):
    seq_len = 7
    pred_len = 7
    sequences = []
    targets_seq = []
    for i in range(len(current_data) - seq_len - pred_len + 1):
        sequences.append(current_data[i:i+seq_len])
        targets_seq.append(current_data[i+seq_len:i+seq_len+pred_len])
    X = torch.tensor(sequences, dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(targets_seq, dtype=torch.float32).to(device)
    model = GRUModel(1, 16, 1, 7).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(5):
        for b in range(0, len(X), 16):
            batch_X = X[b:b+16]
            batch_y = y[b:b+16]
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = torch.tensor(current_data[-7:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).squeeze().tolist()
    forecasts.append(pred[0])
    current_data.append(test_data[step])
print(forecasts)