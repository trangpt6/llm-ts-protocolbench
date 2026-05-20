import pandas as pd
import numpy as np
import torch
import random
from torch.utils.data import DataLoader, TensorDataset
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].values.astype(float)
train_size = 2920
train_data = target[:train_size]
test_data = target[train_size:]
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, pred_len)
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
    return np.array(X), np.array(y)
forecasts = []
current_train = list(train_data)
for i in range(len(test_data)):
    X_train, y_train = create_sequences(np.array(current_train), 7, 7)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    model = GRUModel(1, 16, 1, 7)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for epoch in range(5):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = torch.tensor(current_train[-7:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    with torch.no_grad():
        pred = model(last_seq).squeeze().numpy()
    forecasts.append(float(pred[0]))
    current_train.append(test_data[i])
print(forecasts)