import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# read dataset
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
df['Month'] = pd.to_datetime(df['Month'])
df = df.sort_values('Month').reset_index(drop=True)

# target and feature columns
target_col = 'Ice cream'
feature_cols = ['Heater', 'Ice cream']

# no preprocessing needed per Turn 1 (structural break preserved, no missing values, no imputation)

# chronological 80/20 split
n_total = len(df)
train_size = int(0.8 * n_total)
test_size = n_total - train_size

train_df = df.iloc[:train_size].copy()
test_df = df.iloc[train_size:].copy()

# hyperparameters
input_size = 2
seq_len = 6
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 8
lr = 0.01

# device fallback
device = torch.device('cpu')

# GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.gru(x, h0)
        out = self.fc(out[:, -1, :])
        return out

# dataset class for sequences
class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_len, pred_len):
        self.data = data
        self.seq_len = seq_len
        self.pred_len = pred_len
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_len]
        y = self.data[idx + self.seq_len:idx + self.seq_len + self.pred_len, -1]
        return torch.FloatTensor(x), torch.FloatTensor(y)

# function to create sequences and train model
def train_model(data_array):
    dataset = TimeSeriesDataset(data_array, seq_len, pred_len)
    if len(dataset) == 0:
        return None
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    for epoch in range(epochs):
        model.train()
        for x_batch, y_batch in loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    return model

# function to make one-step prediction
def predict(model, last_seq):
    model.eval()
    with torch.no_grad():
        x = torch.FloatTensor(last_seq).unsqueeze(0).to(device)
        pred = model(x)
        return pred.cpu().numpy()[0, 0]

# normalize data using training statistics
def normalize(data, mean, std):
    return (data - mean) / (std + 1e-8)

def denormalize(data, mean, std):
    return data * (std + 1e-8) + mean

# prepare initial training data
train_data = train_df[feature_cols].values
test_data = test_df[feature_cols].values

# compute normalization stats from initial training data
train_mean = train_data.mean(axis=0)
train_std = train_data.std(axis=0)

# normalize
train_norm = normalize(train_data, train_mean, train_std)

# rolling forecast with retraining at each step
forecasts = []
current_data = np.vstack([train_norm, np.zeros((test_size, input_size))])

for i in range(test_size):
    # train model on current available data
    model = train_model(current_data[:train_size + i])
    
    if model is None:
        # fallback if not enough data
        if i == 0:
            pred = train_df[target_col].iloc[-1]
        else:
            pred = forecasts[-1]
        forecasts.append(pred)
        # update with actual value (normalized)
        actual = test_data[i]
        current_data[train_size + i] = normalize(actual, train_mean, train_std)
        continue
    
    # get last seq_len observations for prediction
    last_seq = current_data[train_size + i - seq_len:train_size + i]
    
    # make prediction
    pred_norm = predict(model, last_seq)
    pred = denormalize(pred_norm, train_mean[-1], train_std[-1])
    forecasts.append(float(pred))
    
    # update with actual observed value (ground truth enabled)
    actual = test_data[i]
    current_data[train_size + i] = normalize(actual, train_mean, train_std)

print(forecasts)