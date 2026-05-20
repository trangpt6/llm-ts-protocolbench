import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import warnings
warnings.filterwarnings('ignore')

# reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# fixed hyperparameters
input_size = 1
seq_len = 13
pred_len = 4
hidden_size = 32
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

# read data
df = pd.read_csv(r'../../../data/ILINet.csv', parse_dates=['DATE'])
target = df['% WEIGHTED ILI'].values.astype(np.float32)

# split (fixed in Turn 0)
total = len(target)
train_size = int(0.8 * total)  # 1043
train_data = target[:train_size]
test_data = target[train_size:]  # length 261

# GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])  # last time step
        return out

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data)-seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)

def train_model(cumulative_data):
    X, y = create_sequences(cumulative_data, seq_len)
    X = torch.tensor(X, dtype=torch.float32).unsqueeze(-1)  # (N, seq_len, 1)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)   # (N, 1)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = GRUModel(input_size, hidden_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
    return model

def predict_multi(model, last_seq):
    # last_seq is (seq_len, 1) tensor
    model.eval()
    preds = []
    current_seq = last_seq.clone()
    with torch.no_grad():
        for _ in range(pred_len):
            inp = current_seq.unsqueeze(0)  # (1, seq_len, 1)
            out = model(inp)                 # (1, 1)
            pred = out.squeeze().item()
            preds.append(pred)
            # shift window: drop first, append prediction
            current_seq = torch.cat([current_seq[1:], torch.tensor([[pred]])], dim=0)
    return preds

# rolling forecast
cumulative = list(train_data)
forecasts = []

for i in range(len(test_data)):
    # train model on current cumulative data
    model = train_model(np.array(cumulative))
    # last sequence for prediction
    last_seq = torch.tensor(cumulative[-seq_len:], dtype=torch.float32).unsqueeze(-1)
    preds = predict_multi(model, last_seq)
    first_pred = preds[0]
    forecasts.append(first_pred)
    # incorporate true next value into cumulative (ground truth enabled)
    true_next = test_data[i]
    cumulative.append(true_next)

# output the forecast list
print(forecasts)