import torch
import torch.nn as nn
import json
from torch.utils.data import TensorDataset, DataLoader
from statistics import mean

# Load training log from your file (replace 'training_log.json' with your actual filename)
with open('training_log.json', 'r') as f:  # Put your 400+ entry file name here
    log = json.load(f)

# Prepare sequences and targets (normalize dynamically based on log)
sequences = []
targets = []
all_values = []
for entry in log:
    seq = [x for x in entry['sequence'][-25:]]  # Last 25 elements
    all_values.extend(seq)
    sequences.append(torch.FloatTensor(seq))  # Shape: (25,)
    targets.append(entry['actual'])
all_values.extend(targets)
data_min = min(all_values)
data_max = min(max(all_values), 300.0)  # Cap at 300.0

for i in range(len(sequences)):
    seq = sequences[i].tolist()
    sequences[i] = torch.FloatTensor([(x - data_min) / (data_max - data_min) for x in seq])  # Shape: (25,)
    targets[i] = (targets[i] - data_min) / (data_max - data_min)

# Stack sequences and targets
sequences = torch.stack(sequences).unsqueeze(-1)  # Shape: (400+, 25, 1)
targets = torch.FloatTensor(targets).unsqueeze(-1)  # Shape: (400+, 1)

# Create dataset and loader
dataset = TensorDataset(sequences, targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define CNN-LSTM hybrid model
class CNNLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, max_seq_length=25):
        super(CNNLSTMModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.cnn = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(16, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = x.transpose(1, 2)  # [batch, 1, seq_length]
        x = self.cnn(x)  # [batch, 16, seq_length]
        x = self.relu(x)
        x = x.transpose(1, 2)  # [batch, seq_length, 16]
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # [batch, 1]
        return x

# Initialize and train with early stopping
model = CNNLSTMModel(input_size=1, hidden_size=50, num_layers=2, max_seq_length=25)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Validation setup (10% of 400+ ≈ 40 samples)
val_size = len(log) // 10  # ~40
val_log = log[-val_size:]
val_sequences = [torch.FloatTensor([x for x in entry['sequence'][-25:]]).view(1, -1, 1) for entry in val_log]
val_targets = [entry['actual'] for entry in val_log]
val_sequences_normalized = []
for seq in val_sequences:
    seq_list = seq.squeeze().tolist()
    val_sequences_normalized.append(torch.FloatTensor([(x - data_min) / (data_max - data_min) for x in seq_list]).view(1, -1, 1))

# Train with early stopping (30 epochs)
model.train()
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(30):
    total_loss = 0
    for seqs, targets in loader:
        outputs = model(seqs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    # Validation
    val_loss = 0
    with torch.no_grad():
        for val_seq, val_target in zip(val_sequences_normalized, val_targets):
            val_pred = model(val_seq)
            val_loss += criterion(val_pred, torch.FloatTensor([(val_target - data_min) / (data_max - data_min)]).view(-1, 1)).item()
    val_loss /= len(val_targets)

    print(f'Epoch [{epoch + 1}/30], Training Loss: {total_loss / len(loader):.4f}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_cnn_lstm_model.pth')
        print("Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Denormalize and validate accuracy for final model
final_model = CNNLSTMModel(input_size=1, hidden_size=50, num_layers=2, max_seq_length=25)
final_model.load_state_dict(torch.load('best_cnn_lstm_model.pth'))
final_model.eval()

# Predictions on full dataset to compare with LSTM
predictions = []
for seq in sequences:
    pred_normalized = final_model(seq.unsqueeze(0)).item()
    pred = pred_normalized * (data_max - data_min) + data_min
    pred = max(1.0, min(100.0, pred))
    predictions.append(pred)

# Calculate MAE for hybrid and LSTM
mae_hybrid = mean([abs(pred - log[i]['actual']) for i, pred in enumerate(predictions)])
mae_lstm = mean([abs(entry['predicted'] - entry['actual']) for entry in log])
print(f"LSTM MAE: {mae_lstm:.4f}")
print(f"Hybrid MAE: {mae_hybrid:.4f}")

# Accuracy within ±1.0
accuracy_within_one = sum(1 for p, a in zip(predictions, [entry['actual'] for entry in log]) if abs(p - a) <= 1.0) / len(log) * 100
print(f"Final Accuracy (within ±1.0): {accuracy_within_one:.2f}%")