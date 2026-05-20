import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# device fallback
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/AirPassengers.csv')
values = df['Passengers'].values.astype(np.float32)

n = len(values)
train_size = int(0.8 * n)
test_size = n - train_size

train = values[:train_size]
test = values[train_size:]

input_size = 1
seq_len = 12
pred_len = 12
hidden_size = 64
num_layers = 2
epochs = 30
batch_size = 12
lr = 0.001
block_size = 12

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(series, seq_len, pred_len):
    X, y = [], []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i+seq_len])
        y.append(series[i+seq_len:i+seq_len+pred_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

def train_model(series):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten().astype(np.float32)
    X, y = make_sequences(scaled, seq_len, pred_len)
    X_t = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    # reproducibility per training run
    torch.manual_seed(42)
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n_samples = X_t.shape[0]
    model.train()
    for epoch in range(epochs):
        perm = torch.randperm(n_samples)
        for i in range(0, n_samples, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_t[idx]
            yb = y_t[idx]
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
    return model, scaler

def predict_block(model, scaler, history):
    last_seq = history[-seq_len:].astype(np.float32)
    scaled = scaler.transform(last_seq.reshape(-1, 1)).flatten().astype(np.float32)
    x = torch.tensor(scaled, dtype=torch.float32).view(1, seq_len, 1).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x).cpu().numpy().flatten()
    pred = scaler.inverse_transform(pred.reshape(-1, 1)).flatten()
    return pred

forecasts = []
history = train.copy()
start = 0
while start < test_size:
    model, scaler = train_model(history)
    block_pred = predict_block(model, scaler, history)
    take = min(block_size, test_size - start)
    forecasts.extend(block_pred[:take].tolist())
    history = np.concatenate([history, test[start:start+take]])
    start += take

forecasts = [float(v) for v in forecasts]
print(forecasts)