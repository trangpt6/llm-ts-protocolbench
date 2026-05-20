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
# Device fallback to cpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(float)
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, kernel_size, dilations):
        super(TCN, self).__init__()
        self.layers = nn.ModuleList()
        in_ch = input_size
        for d in dilations:
            conv = nn.Conv1d(in_ch, hidden_size, kernel_size, dilation=d, padding=(kernel_size - 1) * d)
            self.layers.append(conv)
            in_ch = hidden_size
        self.fc = nn.Linear(hidden_size, 12)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = torch.relu(layer(x))
        x = x[:, :, -1]
        x = self.fc(x)
        return x
def create_sequences(series, seq_len, pred_len):
    X = []
    y = []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i : i + seq_len])
        y.append(series[i + seq_len : i + seq_len + pred_len])
    return np.array(X), np.array(y)
current_series = list(train_data)
forecasts = []
for step in range(len(test_data)):
    X_train, y_train = create_sequences(current_series, 12, 12)
    if len(X_train) == 0:
        forecasts.append(0.0)
        current_series.append(test_data[step])
        continue
    model = TCN(1, 32, 3, [1, 2, 4, 8]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    X_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model.train()
    for epoch in range(10):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    last_seq = np.array(current_series[-12:]).reshape(1, 12, 1)
    last_seq_t = torch.tensor(last_seq, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq_t).cpu().numpy()[0]
    forecasts.append(float(pred[0]))
    current_series.append(test_data[step])
print(forecasts)