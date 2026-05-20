import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# Read preprocessed data
df = pd.read_csv(r'../../../data/ETTh1.csv')
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)
series = df['OT'].copy()

# Preprocessing: detect days with constant values and interpolate
daily_groups = series.groupby(series.index.date)
faulty_days = [k for k, v in daily_groups if v.nunique() == 1]
for d in faulty_days:
    series.loc[series.index.date == d] = np.nan
series.interpolate(method='linear', inplace=True)

# Train/test split
train_len = 14016
train_series = series.iloc[:train_len].values
full_series = series.values  # entire preprocessed series

input_size = 7
seq_len = 12
hidden_size = 32
num_layers = 1
epochs = 3
batch_size = 32
lr = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        return self.fc(out[:, -1, :])

def create_training_data(values, up_to_idx):
    # values: 1D array of length up_to_idx (at least 18)
    X_list, y_list = [], []
    for t in range(seq_len+6, up_to_idx):
        seq = []
        for j in range(t-seq_len, t):
            feat = values[j-6:j+1].tolist()  # 7 elements
            seq.append(feat)
        X_list.append(seq)
        y_list.append(values[t])
    if len(X_list) == 0:
        return None, None
    return torch.tensor(X_list, dtype=torch.float32), torch.tensor(y_list, dtype=torch.float32).view(-1,1)

def train_model(values_up_to):
    model = GRUModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X, y = create_training_data(values_up_to, len(values_up_to))
    if X is None or len(X) == 0:
        return model  # no training possible
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for _ in range(epochs):
        model.train()
        for batch_X, batch_y in loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    return model

def predict(model, values_up_to):
    # values_up_to includes data up to index i-1
    if len(values_up_to) < seq_len+6:
        return np.full(1, np.nan)
    seq = []
    i = len(values_up_to)
    for j in range(i-seq_len, i):
        feat = values_up_to[j-6:j+1].tolist()
        seq.append(feat)
    X = torch.tensor([seq], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred = model(X).cpu().numpy()[0,0]
    return pred

# Initial training
current_values = full_series[:train_len].copy()
model = train_model(current_values)

forecasts = []
for i in range(train_len, len(full_series)):
    # Predict next value
    pred = predict(model, current_values)
    forecasts.append(float(pred))
    # Update current_values with true value (ground truth is allowed after prediction)
    current_values = np.append(current_values, full_series[i])
    # Retrain model on expanded data
    model = train_model(current_values)

print(forecasts)