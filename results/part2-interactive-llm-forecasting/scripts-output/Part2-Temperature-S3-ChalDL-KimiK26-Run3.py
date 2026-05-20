import random
import numpy as np
import torch
import pandas as pd

# set random seeds for reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# read dataset from local file
df = pd.read_csv(r'../../../data/Temperature.csv')
series = df['Daily minimum temperatures'].values.astype(float)

# chronological 80/20 split
train_size = int(0.8 * len(series))
train_data = series[:train_size]
test_data = series[train_size:]

# fixed hyperparameters
input_size = 1
seq_len = 7
pred_len = 7
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01
device = torch.device('cpu')

class GRUModel(torch.nn.Module):
    def __init__(self):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, pred_len)
    def forward(self, x):
        out, h = self.gru(x)
        return self.fc(h[-1])

model = GRUModel().to(device)
criterion = torch.nn.MSELoss()
history = train_data.tolist()
forecasts = []

for i in range(len(test_data)):
    # create sliding windows from current history
    X_list = []
    y_list = []
    n = len(history)
    for j in range(n - seq_len - pred_len + 1):
        X_list.append(history[j:j+seq_len])
        y_list.append(history[j+seq_len:j+seq_len+pred_len])
    X = torch.tensor(np.array(X_list), dtype=torch.float32).unsqueeze(-1).to(device)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).to(device)
    dataset = torch.utils.data.TensorDataset(X, y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # retrain model on all available history
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
    
    # predict next 7 steps and use the first step for the current test timestamp
    model.eval()
    with torch.no_grad():
        seq = torch.tensor(history[-seq_len:], dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
        pred = model(seq)
        forecasts.append(pred[0, 0].item())
    
    # update history with ground truth
    history.append(float(test_data[i]))

print(forecasts)