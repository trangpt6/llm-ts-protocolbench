import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# load dataset and split chronologically
df = pd.read_csv(r'../../../data/AirPassengers.csv')
train_size = 115
train_series = df['Passengers'].iloc[:train_size].values.astype(float)
test_series = df['Passengers'].iloc[train_size:].values.astype(float)

# define LSTM model architecture
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

# fixed hyperparameters
input_size = 1
seq_len = 12
pred_len = 1
hidden_size = 64
num_layers = 2
epochs = 50
batch_size = 16
lr = 0.001

# device fallback to cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# create sliding windows for training
def create_sequences(data, seq_len):
    xs, ys = [], []
    for i in range(len(data) - seq_len):
        xs.append(data[i:i+seq_len])
        ys.append(data[i+seq_len])
    return np.array(xs), np.array(ys)

X_train, y_train = create_sequences(train_series, seq_len)
X_train = X_train.reshape(-1, seq_len, input_size)
y_train = y_train.reshape(-1, pred_len)

X_train_tensor = torch.from_numpy(X_train).float().to(device)
y_train_tensor = torch.from_numpy(y_train).float().to(device)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# train once on the training set
model.train()
for _ in range(epochs):
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

# recursive one-step-ahead forecasting over the test set without ground truth updates
history = list(train_series)
forecasts = []
model.eval()
with torch.no_grad():
    for _ in range(len(test_series)):
        seq = np.array(history[-seq_len:]).reshape(1, seq_len, input_size)
        seq_tensor = torch.from_numpy(seq).float().to(device)
        pred = model(seq_tensor)
        pred_val = pred.item()
        forecasts.append(pred_val)
        history.append(pred_val)

print(forecasts)