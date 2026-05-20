import random
random.seed(0)
import numpy as np
np.random.seed(0)
import torch
torch.manual_seed(0)
import pandas as pd

df = pd.read_csv(r'../../../data/ILINet.csv')

for col in df.columns:
    if pd.isna(df.at[0, col]):
        df.at[0, col] = df[col].iloc[1:].dropna().iloc[0]

target = df['% WEIGHTED ILI'].values.astype(float)
total = len(target)
train_size = int(0.8 * total)
test_size = total - train_size

seq_len = 13
pred_len = 1
input_size = 1
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out

def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    xs = np.array(xs).reshape(-1, seq_len, input_size)
    ys = np.array(ys).reshape(-1, pred_len)
    return xs, ys

train_series = target[:train_size].copy()
forecasts = []

for t in range(test_size):
    X, y = create_sequences(train_series, seq_len)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_input = torch.FloatTensor(train_series[-seq_len:]).reshape(1, seq_len, input_size).to(device)
        pred = model(x_input).cpu().item()
    forecasts.append(pred)

    true_val = target[train_size + t]
    train_series = np.append(train_series, true_val)

print(forecasts)