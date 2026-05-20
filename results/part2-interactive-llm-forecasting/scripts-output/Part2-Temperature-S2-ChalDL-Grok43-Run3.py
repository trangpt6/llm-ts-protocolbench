import pandas as pd
import numpy as np
import torch
import random
# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
df = pd.read_csv(r'../../../data/Temperature.csv')
target = df['Daily minimum temperatures'].astype(float).values
non_zero = target[target != 0]
median_val = np.median(non_zero)
target = np.where(target == 0, median_val, target)
train_size = 2920
train_data = target[:train_size]
test_data = target[train_size:]
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(GRUModel, self).__init__()
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
        y.append(data[i+seq_len])
    return np.array(X), np.array(y)
forecasts = []
train_list = list(train_data)
for i in range(len(test_data)):
    X_train, y_train = create_sequences(train_list, 7)
    X_train = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    model = GRUModel(1, 16, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.MSELoss()
    for epoch in range(5):
        model.train()
        for j in range(0, len(X_train), 16):
            batch_X = X_train[j:j+16]
            batch_y = y_train[j:j+16]
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    model.eval()
    last_seq = torch.tensor(train_list[-7:], dtype=torch.float32).view(1, 7, 1)
    with torch.no_grad():
        pred = model(last_seq).item()
    forecasts.append(pred)
    train_list.append(test_data[i])
print(forecasts)