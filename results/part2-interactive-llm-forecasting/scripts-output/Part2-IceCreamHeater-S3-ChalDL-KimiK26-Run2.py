import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# read data
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# apply preprocessing from Turn 1
df.loc[df['Month'] == '2011-04-01', 'Ice cream'] = 25
df.loc[df['Month'] == '2016-12-01', 'Ice cream'] = 30

# define train/test split
train_size = 157
test_size = 40
train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# prepare data
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']

# hyperparameters
input_size = 2
seq_len = 6
pred_len = 12
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

# build sequences
def build_sequences(data, seq_len, pred_len):
    X, y = [], []
    for i in range(len(data) - seq_len - pred_len + 1):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len:i+seq_len+pred_len, 1])  # predict Ice cream
    return np.array(X), np.array(y)

# LSTM model
class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# training function
def train_model(model, X_train, y_train, epochs, batch_size, lr):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    dataset = torch.utils.data.TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

# rolling forecast with retraining
device = 'cpu'
forecasts = []
history = train_df.copy()

for t in range(test_size):
    # prepare data from history
    data = history[feature_cols].values
    X, y = build_sequences(data, seq_len, pred_len)
    
    if len(X) == 0:
        break
    
    # train model
    model = LSTMForecaster(input_size, hidden_size, num_layers, pred_len)
    train_model(model, X, y, epochs, batch_size, lr)
    model.eval()
    
    # predict: use last seq_len steps
    last_seq = data[-seq_len:]
    x_input = torch.FloatTensor(last_seq).unsqueeze(0)
    with torch.no_grad():
        pred = model(x_input).numpy().flatten()
    
    # store forecast for this step (first value of prediction)
    forecasts.append(float(pred[0]))
    
    # update history with true value (ground truth enabled)
    new_row = test_df.iloc[t:t+1].copy()
    history = pd.concat([history, new_row], ignore_index=True)

print(forecasts)