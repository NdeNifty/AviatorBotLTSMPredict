import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file
from collections import deque
import os
import json

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=100, num_layers=3, max_seq_length=25):
        super(LSTMModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

# Define paths on persistent disk
MODEL_PATH = '/model/best_lstm_model.pth'
TRAINING_LOG_PATH = '/model/training_log.json'

# Initialize or load the model
max_seq_length = 25
model = LSTMModel(max_seq_length=max_seq_length)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Loaded existing model from persistent disk")
except FileNotFoundError:
    print("Starting with a fresh model")

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.008)
criterion = nn.MSELoss()

# Configuration
min_seq_length = 10
data_buffer = deque(maxlen=1000)
data_min, data_max = 1.0, 100.0  # Updated for expected range (1.0–100.0, covering 95%+ of data)
last_sequence = None
request_count = 0
save_interval = 10

app = Flask(__name__)

# Initialize or load training log
training_log = []
if os.path.exists(TRAINING_LOG_PATH):
    with open(TRAINING_LOG_PATH, 'r') as f:
        training_log = json.load(f)

@app.route('/predict', methods=['POST'])
def predict_and_train():
    global data_min, data_max, last_sequence, request_count

    data = request.json['sequence']
    array_length = len(data)
    if array_length < min_seq_length:
        return jsonify({'error': f'Sequence must be at least {min_seq_length} numbers'}), 400

    seq_length = min(array_length, max_seq_length)
    sequence = data[-seq_length:]

    data_buffer.extend(data)
    # Keep data_min/data_max static for stability—commented out dynamic updates
    # data_min = min(data_min, min(data))
    # data_max = max(data_max, max(data))

    seq_normalized = [(x - data_min) / (data_max - data_min) for x in sequence]
    seq_tensor = torch.FloatTensor(seq_normalized).view(1, seq_length, 1)

    with torch.no_grad():
        pred_normalized = model(seq_tensor).item()
    pred = pred_normalized * (data_max - data_min) + data_min
    pred = max(1.0, min(100.0, pred))  # Clamp predictions to 1.0–100.0

    loss = None
    if last_sequence is not None and len(data) > len(last_sequence):
        actual = data[-1]
        actual_normalized = (actual - data_min) / (data_max - data_min)
        target_tensor = torch.FloatTensor([actual_normalized]).view(-1, 1)

        last_seq_length = min(len(last_sequence), max_seq_length)
        last_seq_normalized = [(x - data_min) / (data_max - data_min) for x in last_sequence[-last_seq_length:]]
        last_seq_tensor = torch.FloatTensor(last_seq_normalized).view(1, last_seq_length, 1)

        output = model(last_seq_tensor)
        loss = criterion(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        request_count += 1
        if request_count >= save_interval:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'Saved model to persistent disk after {request_count} requests')
            request_count = 0

        # Store predicted and actual values
        training_log.append({
            'sequence': sequence,  # Last 25 elements
            'predicted': float(pred),  # Convert to float for JSON
            'actual': float(actual)
        })
        with open(TRAINING_LOG_PATH, 'w') as f:
            json.dump(training_log, f)

        # Log high values for monitoring
        if actual > 100.0:
            print(f"High value detected - Actual: {actual:.4f}, Predicted: {pred:.4f}")

        print(
            f'Trained - Seq Length: {last_seq_length}, Predicted: {pred:.4f}, Actual: {actual:.4f}, Loss: {loss.item():.4f}')

    last_sequence = data.copy()
    return jsonify({'prediction': pred, 'loss': loss.item() if loss is not None else None})

@app.route('/download-model', methods=['GET'])
def download_model():
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True, download_name='best_lstm_model.pth')
    else:
        return jsonify({'error': 'Model file not found. Train the model first.'}), 404

@app.route('/download-log', methods=['GET'])
def download_log():
    if os.path.exists(TRAINING_LOG_PATH):
        return send_file(TRAINING_LOG_PATH, as_attachment=True, download_name='training_log.json')
    else:
        return jsonify({'error': 'Training log not found. Train the model first.'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# Summary of Endpoints:
# /predict (POST): Receives a sequence of numbers, predicts the next value, trains the model if the sequence extends the last one,
#                  returns a JSON with the prediction and loss (if trained). Saves model every 10 training steps to persistent disk.
# /download-model (GET): Downloads the trained model file (best_lstm_model.pth) from the persistent disk if it exists,
#                       returns a JSON error if not found.
# /download-log (GET): Downloads the training log file (training_log.json) containing sequence, predicted, and actual values,
#                     returns a JSON error if not found.