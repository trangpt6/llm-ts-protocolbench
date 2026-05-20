import random
import numpy as np
import pandas as pd
import torch

# Set random seeds for reproducibility
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# Device fallback
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 1
seq_len = 7
pred_len = 1
hidden_size = 16
num_layers = 1
epochs = 5
batch_size = 16
lr = 0.01

df = pd.read_csv(r'../../../data/Temperature.csv')
df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
df = df.sort_values("Date").reset_index(drop=True)

train_df = df.iloc[:2920].copy()
test_df = df.iloc[2920:3650].copy()

series_history = train_df["Daily minimum temperatures"].astype(float).to_numpy().tolist()
test_values = test_df["Daily minimum temperatures"].astype(float).to_numpy().tolist()

class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, pred_len):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = torch.nn.Linear(hidden_size, pred_len)

    def forward(self, x):
        out, _ = self.gru(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def make_sequences(values, seq_len):
    x_list = []
    y_list = []
    arr = np.asarray(values, dtype=np.float32)
    for i in range(len(arr) - seq_len):
        x_list.append(arr[i:i + seq_len])
        y_list.append(arr[i + seq_len])
    x = np.asarray(x_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y = np.asarray(y_list, dtype=np.float32).reshape(-1, 1)
    return x, y

def train_and_predict(values):
    torch.manual_seed(42)
    x_np, y_np = make_sequences(values, seq_len)
    x_tensor = torch.tensor(x_np, dtype=torch.float32)
    y_tensor = torch.tensor(y_np, dtype=torch.float32)
    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    generator = torch.Generator()
    generator.manual_seed(42)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)
    model = GRUModel(input_size, hidden_size, num_layers, pred_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
    model.eval()
    last_seq = np.asarray(values[-seq_len:], dtype=np.float32).reshape(1, seq_len, 1)
    with torch.no_grad():
        pred = model(torch.tensor(last_seq, dtype=torch.float32).to(device)).cpu().numpy().reshape(-1)[0]
    return float(pred)

forecasts = []
for true_value in test_values:
    forecast = train_and_predict(series_history)
    forecasts.append(forecast)
    series_history.append(float(true_value))

print(forecasts)