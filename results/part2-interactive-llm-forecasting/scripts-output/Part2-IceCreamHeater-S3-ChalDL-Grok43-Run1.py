import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
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
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 1])
    return np.array(X), np.array(y)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)
train_size = 158
train_data = data[:train_size]
test_data = data[train_size:]
history = train_data.copy()
forecasts = []
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Fallback to CPU if CUDA unavailable
for t in range(len(test_data)):
    X, y = create_sequences(history, 6, 12)
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)
    model = LSTMModel(2, 16, 1, 12).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    model.train()
    for epoch in range(5):
        for i in range(0, len(X_t), 8):
            batch_x = X_t[i:i+8]
            batch_y = y_t[i:i+8]
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = history[-6:].reshape(1, 6, 2)
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq_t)
    next_pred = pred[0, 0].item()
    forecasts.append(next_pred)
    next_row = test_data[t].reshape(1, 2)
    history = np.vstack((history, next_row))
print(forecasts)