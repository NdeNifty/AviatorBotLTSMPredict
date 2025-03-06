# train_locally.py
import torch
import numpy as np
import os
from app.model import HybridCNNLSTMModel
from app.data_utils import max_seq_length, data_min, data_max

# Define MODEL_PATH relative to the project
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_lstm_model.pth')
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Generate synthetic data
def generate_synthetic_data(num_samples=1000):
    sequences = []
    targets = []
    for _ in range(num_samples):
        sequence = np.random.uniform(1.0, 100.0, max_seq_length).tolist()
        target = sequence[-1] + np.random.normal(0, 1.0)
        sequences.append(sequence)
        targets.append(target)
    return sequences, targets

# Initialize model
model = HybridCNNLSTMModel(max_seq_length=max_seq_length)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.HuberLoss()

# Generate data
sequences, targets = generate_synthetic_data()

# Train the model
model.train()
for epoch in range(10):
    total_loss = 0
    for seq, target in zip(sequences, targets):
        seq_tensor = torch.FloatTensor([(x - data_min) / (data_max - data_min) for x in seq]).unsqueeze(0).unsqueeze(0)
        target_tensor = torch.FloatTensor([target]).unsqueeze(0)

        optimizer.zero_grad()
        outputs = model(seq_tensor)
        predicted = outputs['predicted_multiplier']
        loss = criterion(predicted, target_tensor)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    print(f"Epoch {epoch+1}, Loss: {total_loss / len(sequences):.4f}")

# Save the model
torch.save(model.state_dict(), MODEL_PATH)
print(f"Model saved locally at {MODEL_PATH}")