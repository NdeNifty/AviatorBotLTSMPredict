import torch
import torch.nn as nn
import json
from torch.utils.data import TensorDataset, DataLoader

# Load training log from training_log.json
with open('training_log.json', 'r') as f:
    log = json.load(f)

# Prepare sequences and targets (normalize dynamically based on log)
sequences = []
targets = []
all_values = []
for entry in log:
    seq = [x for x in entry['sequence'][-25:]]  # Last 25 elements (max_seq_length=25)
    all_values.extend(seq)
    sequences.append(torch.FloatTensor(seq))  # Shape: (25,)
    targets.append(entry['actual'])
all_values.extend(targets)
data_min = min(all_values)
data_max = min(max(all_values), 300.0)  # Cap at 300.0 for stability

for i in range(len(sequences)):
    seq = sequences[i].tolist()
    sequences[i] = torch.FloatTensor([(x - data_min) / (data_max - data_min) for x in seq])  # Shape: (25,)
    targets[i] = (targets[i] - data_min) / (data_max - data_min)

# Stack sequences and targets into batches (3D for LSTM: batch_size, sequence_length, input_size)
sequences = torch.stack(sequences).unsqueeze(-1)  # Shape: (num_sequences, 25, 1)
targets = torch.FloatTensor(targets).unsqueeze(-1)  # Shape: (num_sequences, 1)

# Create dataset and loader
dataset = TensorDataset(sequences, targets)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

# Define model
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, max_seq_length=25):
        super(LSTMModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)  # x shape: (batch_size, sequence_length, input_size)
        x = self.fc(x[:, -1, :])  # Take last hidden state
        return x

# Initialize and train with early stopping
model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, max_seq_length=25)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Validation setup
val_size = len(log) // 10
val_log = log[-val_size:]
val_sequences = [torch.FloatTensor([x for x in entry['sequence'][-25:]]).view(1, -1, 1) for entry in val_log]
val_targets = [entry['actual'] for entry in val_log]
val_sequences_normalized = []
for seq in val_sequences:
    seq_list = seq.squeeze().tolist()
    val_sequences_normalized.append(torch.FloatTensor([(x - data_min) / (data_max - data_min) for x in seq_list]).view(1, -1, 1))

# Train with early stopping
model.train()
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(50):  # Allow up to 50 for flexibility
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

    print(f'Epoch [{epoch + 1}/50], Training Loss: {total_loss / len(loader):.4f}, Validation Loss: {val_loss:.4f}')

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_lstm_model.pth')
        print("Saved best model")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

# Denormalize and validate accuracy for final model
final_model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, max_seq_length=25)
final_model.load_state_dict(torch.load('best_lstm_model.pth'))
final_model.eval()

predictions = []
for seq in val_sequences_normalized:
    pred_normalized = final_model(seq).item()
    pred = pred_normalized * (data_max - data_min) + data_min
    pred = max(1.0, min(100.0, pred))  # Clamp to 1.0–100.0
    predictions.append(pred)

accuracy_within_one = sum(1 for p, a in zip(predictions, val_targets) if abs(p - a) <= 1.0) / len(val_targets) * 100
print(f"Final Validation Accuracy (within ±1.0): {accuracy_within_one:.2f}%")