import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
data = df[['Heater', 'Ice cream']].values.astype(np.float32)
train_size = 158
train_data = data[:train_size]
test_data = data[train_size:]
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])
        return out
input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01
def create_sequences(data, seq_len, pred_len):
    X = []
    y = []
    for j in range(len(data) - seq_len - pred_len + 1):
        X.append(data[j:j+seq_len])
        y.append(data[j+seq_len:j+seq_len+pred_len, 1])
    return np.array(X), np.array(y)
forecasts = []
current_train = train_data.copy()
for i in range(len(test_data)):
    X_train, y_train = create_sequences(current_train, seq_len, pred_len)
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len)
    # Fallback to CPU if CUDA unavailable
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    X_train = X_train.to(device)
    y_train = y_train.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    model.train()
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = current_train[-seq_len:]
    last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq)
        pred = pred.cpu().squeeze().numpy()
    forecasts.append(float(pred[0]))
    current_train = np.vstack((current_train, test_data[i]))
print(forecasts)