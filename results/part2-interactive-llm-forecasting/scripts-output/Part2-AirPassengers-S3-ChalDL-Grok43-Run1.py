import pandas as pd
import numpy as np
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/AirPassengers.csv')
data = df['Passengers'].values.astype(np.float32)
train_size = 115
train_data = data[:train_size]
test_data = data[train_size:]
class TCNModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, pred_len):
        super(TCNModel, self).__init__()
        self.conv_layers = torch.nn.ModuleList()
        for i in range(num_layers):
            dilation = dilations[i]
            in_ch = input_size if i == 0 else hidden_size
            self.conv_layers.append(torch.nn.Conv1d(in_ch, hidden_size, kernel_size, dilation=dilation, padding=(kernel_size - 1) * dilation))
            self.conv_layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        for layer in self.conv_layers:
            x = layer(x)
        x = x[:, :, -1]
        x = self.fc(x)
        return x
def create_sequences(series, seq_len, pred_len):
    X = []
    y = []
    for i in range(len(series) - seq_len - pred_len + 1):
        X.append(series[i:i + seq_len])
        y.append(series[i + seq_len:i + seq_len + pred_len])
    return np.array(X), np.array(y)
def train_model(model, optimizer, criterion, series, seq_len, pred_len, epochs, batch_size):
    X, y = create_sequences(series, seq_len, pred_len)
    if len(X) < batch_size:
        return
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)
    y = torch.tensor(y, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for _ in range(epochs):
        for batch_X, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
hyperparams = {'input_size': 1, 'seq_len': 12, 'pred_len': 12, 'hidden_size': 32, 'num_layers': 2, 'epochs': 10, 'batch_size': 8, 'lr': 0.01, 'kernel_size': 3, 'dilations': [1, 2, 4, 8]}
model = TCNModel(hyperparams['input_size'], hyperparams['hidden_size'], hyperparams['num_layers'], hyperparams['kernel_size'], hyperparams['dilations'], hyperparams['pred_len'])
optimizer = torch.optim.Adam(model.parameters(), lr=hyperparams['lr'])
criterion = torch.nn.MSELoss()
current_series = list(train_data)
forecasts = []
for i in range(len(test_data)):
    train_model(model, optimizer, criterion, np.array(current_series), hyperparams['seq_len'], hyperparams['pred_len'], hyperparams['epochs'], hyperparams['batch_size'])
    last_window = np.array(current_series[-hyperparams['seq_len']:]).reshape(1, hyperparams['seq_len'], 1)
    last_window = torch.tensor(last_window, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        pred = model(last_window).cpu().numpy()[0]
    forecasts.append(float(pred[0]))
    current_series.append(test_data[i])
print(forecasts)