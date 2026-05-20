import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# fixed hyperparameters
seq_len = 48
pred_len = 1
input_size = 7
hidden_size = 64
num_layers = 3
kernel_size = 3
dilations = [1, 2, 4, 8, 16]
epochs = 30
batch_size = 64
lr = 0.001

# read data
df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
values = df['OT'].values.astype(np.float32)
n = len(values)
train_size = int(0.8 * n)
train_vals = values[:train_size]
test_vals = values[train_size:]

# create lag features: for each time t, feature vector = [OT(t-1), ..., OT(t-7)]
def create_lags(series, lag=7):
    data = []
    for i in range(lag, len(series)):
        data.append(series[i-lag:i])
    return np.array(data)

# training data preparation
train_lags = create_lags(train_vals, lag=input_size)  # shape (num_samples, 7)
# now create sequences of length seq_len
X_train = []
y_train = []
for i in range(len(train_lags) - seq_len):
    X_train.append(train_lags[i:i+seq_len])   # (seq_len, 7)
    y_train.append(train_vals[input_size+seq_len+i]) # target is the next value after the window
X_train = np.array(X_train, dtype=np.float32)  # (N, seq_len, 7)
y_train = np.array(y_train, dtype=np.float32)  # (N,)
train_dataset = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# TCN implementation (simple version)
class TCN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, kernel_size, dilations, seq_len):
        super(TCN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dilations = dilations
        layers = []
        for i in range(num_layers):
            dilation = dilations[i % len(dilations)]
            padding = (kernel_size - 1) * dilation
            conv = nn.Conv1d(input_size if i == 0 else hidden_size,
                             hidden_size,
                             kernel_size,
                             padding=padding,
                             dilation=dilation)
            layers.append(conv)
            layers.append(nn.ReLU())
        self.conv_layers = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_size * seq_len, 1)  # flatten last dimension

    def forward(self, x):
        # x shape: (batch, seq_len, input_size)
        x = x.permute(0, 2, 1)  # to (batch, input_size, seq_len)
        out = self.conv_layers(x)
        # out shape: (batch, hidden_size, L) where L may be longer due to padding
        # take last seq_len elements
        out = out[:, :, -seq_len:]  # trim to seq_len
        out = out.reshape(out.size(0), -1)  # flatten
        return self.fc(out).squeeze()

model = TCN(input_size, hidden_size, num_layers, kernel_size, dilations, seq_len).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    # print(f'Epoch {epoch+1}, loss {total_loss/len(train_loader):.6f}')

# recursive forecasting on test set
model.eval()
# we need to maintain a buffer of the last seq_len+input_size true values initially from training
# we have train_vals, we'll use the last input_size lags from the training set and the last seq_len true values
# to build the first window.
# We'll create a buffer that holds the last 'seq_len' vectors of 7 lags.
# Start by constructing the first feature vector for test index 0 using true lags from training.
# Actually, we can compute the first test feature from the last 7 lags of training, then build windows step by step.
# We'll use a sliding window over the actual test true values? No, we cannot use ground truth after prediction.

# Method: maintain a list of past true values (from training) that grows with predictions.
past_true = list(train_vals)  # start with all training values
predictions = []
for i in range(len(test_vals)):
    # compute the feature vector for the current time point (which is len(past_true) index)
    # feature = past_true[-7:] (7 lags)
    if len(past_true) < 7:
        # insufficient lags, pad with zeros (but should not happen)
        feat = np.zeros(7)
    else:
        feat = past_true[-7:]
    # now the window = last seq_len feature vectors (including current one)
    # we need seq_len previous feature vectors. But we have only one new feature per step.
    # We need to reconstruct the window from the past features. We can store all feature vectors as we go.
    # Simpler: we can store a list of feature vectors computed at each step (both true and predicted).
    # We'll maintain a list 'feat_buffer' that holds the last seq_len feature vectors.
    # Initially, we compute feature vectors for the last seq_len time points in the training set.
    # To do this robustly, we'll first compute all feature vectors for the entire training set.
    # Then for forecasting, we build the window from the last seq_len feature vectors.
    # This approach is easier.
    pass

# Better approach: Precompute all training feature vectors (lags) and then simulate.
# Reinitialize:
all_train_lags = create_lags(train_vals, lag=input_size)  # shape (train_size-7, 7)
# The last feature vector corresponds to time index train_size-1 (since lags up to there)
# We need seq_len feature vectors to form a window.
# For the first test prediction, we use the last seq_len training feature vectors.
# Then after each prediction, we compute the feature vector from the newly predicted value (plus previous true/predicted values) and append to a buffer.

# Let's do it step by step.
window = list(all_train_lags[-seq_len:])  # list of numpy arrays of length 7, length seq_len
predictions = []
# we also need the actual past values for computing future lags (we will use predicted values)
# we maintain a list of past true values (training) and then predicted values
past_vals = list(train_vals)  # all training values

for i in range(len(test_vals)):
    # current window is of length seq_len, each element is a 7-dim vector
    # prepare input tensor
    x = np.array(window, dtype=np.float32)  # (seq_len, 7)
    x = torch.from_numpy(x).unsqueeze(0).to(device)  # (1, seq_len, 7)
    with torch.no_grad():
        pred = model(x).cpu().numpy()[0]
    predictions.append(float(pred))
    # now update past_vals and window
    past_vals.append(pred)
    # compute new feature from the last 7 values in past_vals
    new_feat = np.array(past_vals[-7:], dtype=np.float32)
    # remove oldest feature from window and append new one
    window.pop(0)
    window.append(new_feat)

# now predictions list has length 3485
print(predictions)