import random
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_csv(r'../../../data/ETTh1.csv', parse_dates=['date'])
df.sort_values('date', inplace=True)
df.reset_index(drop=True, inplace=True)

# time features
df['hour'] = df['date'].dt.hour
df['dayofweek'] = df['date'].dt.dayofweek
df['month'] = df['date'].dt.month

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# select features and target
feature_cols = ['OT', 'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos']
target_col = 'OT'

X = df[feature_cols].values
y = df[target_col].values.reshape(-1, 1)

total_timesteps = len(df)
train_size = int(0.8 * total_timesteps)
test_size = total_timesteps - train_size

# scaling
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y).flatten()

# parameters
seq_len = 12
pred_len = 24
input_size = 7
hidden_size = 32
num_layers = 1
batch_size = 32
lr = 0.005
epochs = 3

# create sequences for training
def create_sequences(X, y, seq_len, pred_len):
    X_seq, y_seq = [], []
    for i in range(seq_len - 1, len(X) - pred_len):
        X_seq.append(X[i - seq_len + 1 : i + 1])
        y_seq.append(y[i + 1 : i + 1 + pred_len])
    return np.array(X_seq), np.array(y_seq)

train_X_seq, train_y_seq = create_sequences(X_scaled[:train_size], y_scaled[:train_size], seq_len, pred_len)

# LSTM model
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# training function
def train_model(model, X, y, batch_size, epochs):
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.train()
    for _ in range(epochs):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            output = model(bx)
            loss = criterion(output, by)
            loss.backward()
            optimizer.step()

train_model(model, train_X_seq, train_y_seq, batch_size, epochs)

# rolling forecast
forecasts = []
# full scaled array for incremental update
current_X_scaled = X_scaled.copy()
current_y_scaled = y_scaled.copy()

for step in range(test_size):
    # last seq_len observations before the target
    start_idx = train_size + step - seq_len + 1
    end_idx = train_size + step
    input_seq = current_X_scaled[start_idx:end_idx]  # (seq_len, 7)
    input_tensor = torch.FloatTensor(input_seq).unsqueeze(0).to(device)  # (1, seq_len, 7)
    model.eval()
    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy().flatten()  # (24,)
    # take first predicted value
    pred_first_scaled = pred_scaled[0]
    pred_first = scaler_y.inverse_transform([[pred_first_scaled]])[0, 0]
    forecasts.append(pred_first)
    # after prediction, true value becomes available
    true_val = y[train_size + step]
    # update scaled arrays (already have true scaled)
    # no need to update, we just need to retrain on all data up to train_size+step+1
    # retrain on entire seen data (including the new true)
    # we reuse the training function with all data up to current point
    new_train_X_seq, new_train_y_seq = create_sequences(
        current_X_scaled[:train_size + step + 1],
        current_y_scaled[:train_size + step + 1],
        seq_len, pred_len
    )
    # reset model weights? No, we continue from previous weights (fine-tune)
    # reinitialize model and optimizer? The setup says retrain at every time step.
    # For simplicity, we reinitialize the model and optimizer each time.
    model = LSTMModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_model(model, new_train_X_seq, new_train_y_seq, batch_size, epochs)

print(forecasts)