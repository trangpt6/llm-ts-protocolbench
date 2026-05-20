import pandas as pd
import numpy as np
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target_col = 'Daily minimum temperatures'
non_zero = df[target_col][df[target_col] != 0]
median_nonzero = non_zero.median()
df[target_col] = df[target_col].replace(0, median_nonzero)
train_size = 2920
train = df[target_col].iloc[:train_size].values.astype(float)
test = df[target_col].iloc[train_size:].values.astype(float)
class LSTMForecaster(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out
input_size = 1
seq_len = 14
pred_len = 30
hidden_size = 64
num_layers = 2
epochs = 20
batch_size = 32
lr = 0.001
def create_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len])
    return np.array(X), np.array(y)
forecasts = []
current_data = train.copy()
i = 0
while i < len(test):
    X_train, y_train = create_sequences(current_data, seq_len, pred_len)
    if len(X_train) == 0:
        break
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    model.train()
    for epoch in range(epochs):
        for j in range(0, len(X_train), batch_size):
            batch_x = X_train[j:j+batch_size]
            batch_y = y_train[j:j+batch_size]
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = current_data[-seq_len:]
    last_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).squeeze().numpy()
    remaining = len(test) - i
    take = min(pred_len, remaining)
    forecasts.extend(pred[:take].tolist())
    block_true = test[i:i+take]
    current_data = np.concatenate([current_data, block_true])
    i += take
print(forecasts)