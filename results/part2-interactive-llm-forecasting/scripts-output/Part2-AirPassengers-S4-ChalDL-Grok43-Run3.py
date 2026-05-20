import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
def create_sequences(data, seq_len, pred_len):
    X = []
    y = []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
device = torch.device('cpu')
current_train = train_data.copy()
forecasts = []
test_idx = 0
seq_len = 12
pred_len = 12
while test_idx < len(test_data):
    X, y = create_sequences(current_train, seq_len, pred_len)
    if len(X) == 0:
        break
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
    model = LSTMModel(1, 64, 2, 12).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=12, shuffle=True)
    for epoch in range(30):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = current_train[-12:].reshape(1, 12, 1)
    last_seq_tensor = torch.tensor(last_seq, dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(last_seq_tensor).cpu().numpy().flatten()
    remaining = len(test_data) - test_idx
    take = min(12, remaining)
    forecasts.extend(pred[:take].tolist())
    true_to_append = test_data[test_idx:test_idx + take]
    current_train = np.concatenate([current_train, true_to_append])
    test_idx += take
print(forecasts)