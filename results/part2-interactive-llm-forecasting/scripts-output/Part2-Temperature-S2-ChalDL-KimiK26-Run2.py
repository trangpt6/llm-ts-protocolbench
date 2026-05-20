import random
import numpy as np
import torch
import pandas as pd

# reproducibility
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)

# device fallback to cpu if cuda unavailable
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load dataset
df = pd.read_csv(r'../../../data/Temperature.csv')
values = df['Daily minimum temperatures'].values.astype(np.float32)
values = values.reshape(-1, 1)

# chronological 80/20 split
train_size = int(0.8 * len(values))
train_data = values[:train_size]
test_data = values[train_size:]

# fixed hyperparameters
input_size = 1
seq_len = 7
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def create_sequences(data, seq_len, pred_len):
    xs = []
    ys = []
    for i in range(len(data) - seq_len - pred_len + 1):
        x = data[i:i+seq_len]
        y = data[i+seq_len:i+seq_len+pred_len]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

forecasts = []
history = train_data.copy()

for t in range(len(test_data)):
    X, y = create_sequences(history, seq_len, pred_len)
    X_tensor = torch.FloatTensor(X)
    y_tensor = torch.FloatTensor(y)
    dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.train()
    for epoch in range(epochs):
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        x_input = history[-seq_len:].reshape(1, seq_len, input_size)
        x_input = torch.FloatTensor(x_input).to(device)
        pred = model(x_input).cpu().numpy().flatten()[0]

    forecasts.append(float(pred))
    history = np.vstack([history, test_data[t].reshape(1, 1)])

print(forecasts)