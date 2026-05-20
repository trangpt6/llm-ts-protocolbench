import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Use CPU to ensure compatibility across environments
device = torch.device('cpu')

# Fixed hyperparameters from Turn 2
input_size = 1
seq_len = 7
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# Load dataset from local CSV
df = pd.read_csv(r'../../../data/Temperature.csv')
values = df['Daily minimum temperatures'].values.astype(np.float32)
n_total = len(values)
train_size = int(0.8 * n_total)
test_size = n_total - train_size

train_data = values[:train_size]
test_data = values[train_size:]

class GRUModel(nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(data, seq_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len):
        x = data[i:i + seq_len]
        y = data[i + seq_len]
        xs.append(x)
        ys.append(y)
    X = np.array(xs).reshape(-1, seq_len, input_size)
    y = np.array(ys).reshape(-1, pred_len)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

def train_model(model, X, y):
    model.train()
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

def predict(model, x):
    model.eval()
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
    return pred.cpu().numpy()[0, 0]

forecasts = []
current_data = train_data.copy()

# Initial training on the chronological train set
X_train, y_train = create_sequences(current_data, seq_len)
model = GRUModel().to(device)
train_model(model, X_train, y_train)

# Rolling one-step ahead forecast with retraining after each observation
for i in range(test_size):
    seq = current_data[-seq_len:].reshape(1, seq_len, input_size)
    seq_tensor = torch.tensor(seq, dtype=torch.float32)
    pred = predict(model, seq_tensor)
    forecasts.append(float(pred))
    current_data = np.append(current_data, test_data[i])
    X_train, y_train = create_sequences(current_data, seq_len)
    model = GRUModel().to(device)
    train_model(model, X_train, y_train)

print(forecasts)