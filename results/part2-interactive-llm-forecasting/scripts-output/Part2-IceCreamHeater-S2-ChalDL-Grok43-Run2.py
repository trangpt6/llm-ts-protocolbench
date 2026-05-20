import pandas as pd
import numpy as np
import torch
import random
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df = pd.read_csv(r'../../../data/IceCreamHeater.csv')
features = df[['Heater', 'Ice cream']].values.astype(np.float32)
train_size = 158
class GRUForecaster(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.gru(x)
        out = self.fc(out[:, -1, :])
        return out
def create_sequences(data, seq_len):
    X = []
    y = []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i + seq_len, 1])
    return np.array(X), np.array(y)
def train_model(model, X, y, epochs, batch_size, lr):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
    dataset = torch.utils.data.TensorDataset(X_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()
    return model
forecasts = []
current_features = features[:train_size].copy()
for step in range(40):
    X_train, y_train = create_sequences(current_features, 6)
    model = GRUForecaster(2, 16, 1)
    model = train_model(model, X_train, y_train, 5, 8, 0.01)
    last_seq = torch.tensor(current_features[-6:].reshape(1, 6, 2), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(last_seq).item()
    forecasts.append(pred)
    next_true = features[train_size + step]
    current_features = np.vstack([current_features, next_true])
print(forecasts)