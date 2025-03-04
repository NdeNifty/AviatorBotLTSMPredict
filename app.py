import torch
import torch.nn as nn
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from collections import deque
import os
import json
from statistics import mean, stdev

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, max_seq_length=25):
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
LOSS_HISTORY_PATH = '/model/loss_history.json'

# Initialize or load the model
max_seq_length = 25
model = LSTMModel(max_seq_length=max_seq_length)
try:
    model.load_state_dict(torch.load(MODEL_PATH))
    print("Loaded existing model from persistent disk")
except FileNotFoundError:
    try:
        model.load_state_dict(torch.load('best_lstm_model.pth'))
        print("Loaded model from GitHub file, saving to persistent disk")
        torch.save(model.state_dict(), MODEL_PATH)
    except FileNotFoundError:
        print("Starting with a fresh model")

model.train()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.HuberLoss(delta=3.0)  # Updated to delta=3.0 for stricter error handling as requested
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

# Configuration
min_seq_length = 10
data_buffer = deque(maxlen=1000)
data_min, data_max = 1.0, 100.0
last_sequence = None
request_count = 0
save_interval = 10
loss_history = []

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/performance": {"origins": "https://aviatorbotltsmpredict.onrender.com"}, r"/training-log": {"origins": "https://aviatorbotltsmpredict.onrender.com"}})

def load_or_init_training_log():
    global training_log
    if os.path.exists(TRAINING_LOG_PATH):
        try:
            with open(TRAINING_LOG_PATH, 'r') as f:
                training_log = json.load(f)
            print(f"Loaded training log with {len(training_log)} entries")
            if len(training_log) > 1000:
                training_log = training_log[-1000:]
        except json.JSONDecodeError:
            print("Corrupted training log, starting fresh with empty list")
            training_log = []
    else:
        print("No training log found, starting fresh with empty list")
        training_log = []
    return training_log

def load_or_init_loss_history():
    global loss_history
    if os.path.exists(LOSS_HISTORY_PATH):
        try:
            with open(LOSS_HISTORY_PATH, 'r') as f:
                loss_history = json.load(f)
            print(f"Loaded loss history with {len(loss_history)} entries")
            if len(loss_history) > 1000:
                loss_history = loss_history[-1000:]
        except json.JSONDecodeError:
            print("Corrupted loss history, starting fresh with empty list")
            loss_history = []
    else:
        print("No loss history found, starting fresh with empty list")
        loss_history = []
    return loss_history

training_log = load_or_init_training_log()
loss_history = load_or_init_loss_history()

def calculate_performance(log):
    if not log:
        return {
            "mae": 0.0,
            "within_one_percent": 0.0,
            "high_value_count": 0,
            "low_value_count": 0,
            "predicted_actual": []  # Keep for existing charts, but we’ll use /training-log for third chart
        }
    
    mae = sum(abs(entry["predicted"] - entry["actual"]) for entry in log) / len(log)
    within_one = sum(1 for entry in log if abs(entry["predicted"] - entry["actual"]) <= 1.0) / len(log) * 100
    high_value_count = sum(1 for entry in log if entry["actual"] > 100.0)
    low_value_count = sum(1 for entry in log if entry["actual"] <= 100.0)
    
    # Collect predicted vs. actual pairs (last 250 for memory, optional for /performance)
    predicted_actual = [(entry["predicted"], entry["actual"]) for entry in log[-250:]]
    
    return {
        "mae": float(mae),
        "within_one_percent": float(within_one),
        "high_value_count": high_value_count,
        "low_value_count": low_value_count,
        "predicted_actual": predicted_actual
    }

@app.route('/predict', methods=['POST'])
def predict_and_train():
    global data_min, data_max, last_sequence, request_count, training_log, loss_history

    data = request.json['sequence']
    array_length = len(data)
    if array_length < min_seq_length:
        return jsonify({'error': f'Sequence must be at least {min_seq_length} numbers'}), 400

    seq_length = min(array_length, max_seq_length)
    sequence = data[-seq_length:]

    data_buffer.extend(data)
    data_min = min(data_min, min(data))
    data_max = min(max(data_max, max(data)), 300.0)

    seq_normalized = [(x - data_min) / (data_max - data_min) for x in sequence]
    seq_tensor = torch.FloatTensor(seq_normalized).view(1, seq_length, 1)

    with torch.no_grad():
        pred_normalized = model(seq_tensor).item()
    pred = pred_normalized * (data_max - data_min) + data_min
    pred = max(1.0, min(100.0, pred))

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
        scheduler.step(loss)

        request_count += 1
        if request_count >= save_interval:
            torch.save(model.state_dict(), MODEL_PATH)
            print(f'Saved model to persistent disk after {request_count} requests')
            request_count = 0

        loss_history.append(float(loss.item()))
        if len(loss_history) > 1000:
            loss_history = loss_history[-1000:]
        try:
            with open(LOSS_HISTORY_PATH, 'w') as f:
                json.dump(loss_history, f)
            print(f"Updated loss history with {len(loss_history)} entries")
        except (IOError, json.JSONEncodeError) as e:
            print(f"Error saving loss history: {e}")

        training_log.append({
            'sequence': sequence,
            'predicted': float(pred),
            'actual': float(actual)
        })
        try:
            with open(TRAINING_LOG_PATH, 'w') as f:
                json.dump(training_log, f)
            print(f"Updated training log with {len(training_log)} entries")
        except (IOError, json.JSONEncodeError) as e:
            print(f"Error saving training log: {e}. Truncating log to last 1000 entries.")
            training_log = training_log[-1000:]
            with open(TRAINING_LOG_PATH, 'w') as f:
                json.dump(training_log, f)

        if actual > 100.0:
            print(f"High value detected - Actual: {actual:.4f}, Predicted: {pred:.4f}")

        print(f'Trained - Seq Length: {last_seq_length}, Predicted: {pred:.4f}, Actual: {actual:.4f}, Loss: {loss.item():.4f}')

    last_sequence = data.copy()
    return jsonify({'prediction': pred, 'loss': loss.item() if loss is not None else None})

@app.route('/performance', methods=['GET'])
def get_performance():
    global training_log, loss_history
    
    performance = calculate_performance(training_log)
    response = {
        "performance": performance,
        "learning_curve": loss_history
    }
    return jsonify(response)

@app.route('/training-log', methods=['GET'])
def get_training_log():
    if os.path.exists(TRAINING_LOG_PATH):
        try:
            with open(TRAINING_LOG_PATH, 'r') as f:
                log_data = json.load(f)
            if not log_data:
                return jsonify({'error': 'Training log is empty'}), 404
            return jsonify(log_data)
        except (IOError, json.JSONDecodeError) as e:
            print(f"Error reading training log: {e}")
            return jsonify({'error': 'Failed to read training log'}), 500
    else:
        return jsonify({'error': 'Training log not found'}), 404

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

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

@app.route('/download-loss-history', methods=['GET'])
def download_loss_history():
    if os.path.exists(LOSS_HISTORY_PATH):
        return send_file(LOSS_HISTORY_PATH, as_attachment=True, download_name='loss_history.json')
    else:
        return jsonify({'error': 'Loss history not found. Train the model first.'}), 404

@app.route('/upload-model', methods=['POST'])
def upload_model():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename != 'best_lstm_model.pth':
        return jsonify({'error': 'Invalid file name'}), 400
    
    file.save(MODEL_PATH)
    print(f"Overwrote existing model with uploaded best_lstm_model.pth at {MODEL_PATH}")
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        print("Reloaded model from uploaded file")
    except RuntimeError as e:
        return jsonify({'error': f'Failed to load uploaded model: {e}'}), 500
    
    return jsonify({'message': 'Model successfully uploaded and overwrote existing model'})

@app.route('/delete-log', methods=['DELETE'])
def delete_log():
    global training_log
    if os.path.exists(TRAINING_LOG_PATH):
        os.remove(TRAINING_LOG_PATH)
        training_log = []
        print("Training log deleted successfully")
        return jsonify({'message': 'Training log deleted successfully'}), 200
    else:
        return jsonify({'error': 'Training log not found'}), 404

@app.route('/delete-loss-history', methods=['DELETE'])
def delete_loss_history():
    global loss_history
    if os.path.exists(LOSS_HISTORY_PATH):
        os.remove(LOSS_HISTORY_PATH)
        loss_history = []
        print("Loss history deleted successfully")
        return jsonify({'message': 'Loss history deleted successfully'}), 200
    else:
        return jsonify({'error': 'Loss history not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# Summary of Endpoints:
# / (GET): Serves the static index.html for visualization, loading charts for performance metrics, learning curve, and predicted vs. actual payouts.
# /predict (POST): Receives a sequence of numbers, predicts the next value, trains the model if the sequence extends the last one,
#                  returns a JSON with the prediction and loss (if trained). Saves model every 10 training steps to persistent disk
#                  and logs predicted/actual values to training_log.json and loss to loss_history.json.
# /performance (GET): Returns JSON with performance metrics (MAE, % within ±1.0, high/low value counts) and learning curve (loss history).
# /training-log (GET): Returns JSON of the training_log.json file from persistent disk, containing sequence, predicted, and actual values for visualization.
# /download-model (GET): Downloads the trained model file (best_lstm_model.pth) from the persistent disk if it exists,
#                       returns a JSON error if not found.
# /download-log (GET): Downloads the training log file (training_log.json) containing sequence, predicted, and actual values,
#                     returns a JSON error if not found.
# /download-loss-history (GET): Downloads the loss history file (loss_history.json) containing loss values over time,
#                              returns a JSON error if not found.
# /upload-model (POST): Uploads a new best_lstm_model.pth file to overwrite any existing model on the persistent disk,
#                      returns a JSON success message or error if the file is invalid or upload fails.
# /delete-log (DELETE): Deletes the training log file (training_log.json) from the persistent disk if it exists, resets the in-memory training_log.
# /delete-loss-history (DELETE): Deletes the loss history file (loss_history.json) from the persistent disk if it exists, resets the in-memory loss_history.