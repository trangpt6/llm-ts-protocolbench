import random
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Load data
df = pd.read_csv(r'../../../data/Temperature.csv')
series = df['Daily minimum temperatures'].values.astype(np.float32)

# Chronological split
train_size = int(0.8 * len(series))
history = list(series[:train_size])   # mutable list
test = series[train_size:]

# Hyperparameters
input_size = 1
seq_len = 7
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class GRUModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])   # last time step output
        return out.squeeze(-1)

def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(seq_len, len(data)):
        X.append(data[i-seq_len:i])
        y.append(data[i])
    return torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32)

def train_model(train_data):
    model = GRUModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    X_train, y_train = create_sequences(train_data, seq_len)
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    return model

forecasts = []
for i in range(len(test)):
    # Retrain on all data available so far (including previously observed test points)
    model = train_model(history)
    # Use the last seq_len values to predict next step
    last_seq = torch.tensor(history[-seq_len:], dtype=torch.float32).view(1, seq_len, 1).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).item()
    forecasts.append(pred)
    # Update history with the true test value (ground truth)
    history.append(float(test[i]))

print(forecasts)