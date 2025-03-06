from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from .data_utils import initialize_model, data_buffer, data_min, data_max, last_sequence, max_seq_length, training_queue, TRAINING_QUEUE_PATH
from .training_utils import assign_safety_label, assign_rtp_window  # Removed get_performance_metrics, get_training_log
import os
import json
from datetime import datetime
import time

app = Flask(__name__)
CORS(app)  # Simplified CORS since /performance and /training-log are removed

# Initialize device for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model with GPU support
initialize_model()  # This will now use GPU if available (updated in data_utils.py)

# Note: The training loop is already started in training_utils.py, so no additional thread is needed here

# Route to predict the next multiplier
@app.route('/predict', methods=['POST'])
def predict():
    global model
    from .data_utils import data_buffer, data_min, data_max, last_sequence, max_seq_length
    
    prediction_data = request.json.get('predictionData', {})
    sequence = prediction_data.get('Last_30_Multipliers', [])
    time_gaps = prediction_data.get('Time_Gaps', None)  # Optional

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    # Log Time_Gaps for future use
    if time_gaps is not None:
        print(f"Received Time_Gaps: {time_gaps}")
    else:
        print("No Time_Gaps provided")

    # Pad sequence with default value (1.0) if less than 10
    seq_length = min(len(sequence), max_seq_length)
    padded_sequence = sequence[-seq_length:]
    while len(padded_sequence) < 10:
        padded_sequence.insert(0, 1.0)  # Pad with 1.0 at the beginning
    
    data_buffer.extend(padded_sequence)
    data_min = min(data_min, min(padded_sequence))
    data_max = min(max(data_max, max(padded_sequence)), 300.0)

    seq_normalized = [(x - data_min) / (data_max - data_min) for x in padded_sequence]
    seq_tensor = torch.FloatTensor(seq_normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        model.to(device)  # Ensure model is on the correct device
        outputs = model(seq_tensor)
        predicted_multiplier = outputs['predicted_multiplier'].item()
        confidence_score = outputs['confidence_score'].item()
        classifier_out = outputs['classifier_output']
        
        safety_label = assign_safety_label(confidence_score * 100, predicted_multiplier)  # Convert to percentage for your logic
        rtp_window = assign_rtp_window(padded_sequence)
        
        response = {
            'predicted_multiplier': float(predicted_multiplier),
            'confidence_score': float(confidence_score),
            'safety_label': safety_label,
            'RTP_window': rtp_window
        }
        return jsonify(response)

# Route to delete the model file
@app.route('/delete-model', methods=['DELETE'])
def delete_model():
    from .data_utils import MODEL_PATH
    try:
        if os.path.exists(MODEL_PATH):
            os.remove(MODEL_PATH)
            print("Model file deleted successfully")
            return jsonify({'message': 'Model file deleted successfully'})
        else:
            print("Model file not found")
            return jsonify({'error': 'Model file not found'}), 404
    except Exception as e:
        print(f"Error deleting model file: {e}")
        return jsonify({'error': str(e)}), 500

# Route to upload a new model
@app.route('/upload-model', methods=['POST'])
def upload_model():
    from .data_utils import MODEL_PATH
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pth'):
        try:
            file.save(MODEL_PATH)
            state_dict = torch.load(MODEL_PATH, map_location=device)
            from .model import HybridCNNLSTMModel
            global model
            model = HybridCNNLSTMModel(max_seq_length=max_seq_length).to(device)  # Move to GPU
            model.load_state_dict(state_dict, strict=False)
            print("Model uploaded and loaded successfully")
            return jsonify({'message': 'Model uploaded successfully'})
        except Exception as e:
            print(f"Error uploading model: {e}")
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid file format'}), 400

# Route to train the model with new data
@app.route('/train', methods=['POST'])
def train():
    global training_queue
    from .data_utils import TRAINING_QUEUE_PATH
    data = request.json.get('loggingData', {})
    if not data:
        return jsonify({'error': 'No logging data provided'}), 400

    # Add actual multiplier if not provided (assuming prediction outcome)
    if 'Actual_Multiplier' not in data:
        data['Actual_Multiplier'] = data.get('Multiplier_Outcome', 1.0)  # Fallback to latest multiplier

    training_queue.put({'loggingData': data, 'timestamp': datetime.utcnow().isoformat()})
    try:
        with open(TRAINING_QUEUE_PATH, 'w') as f:
            json.dump(list(training_queue.queue), f)
        print(f"Added to training queue, total: {training_queue.qsize()}")
    except (IOError, json.JSONEncodeError) as e:
        print(f"Error saving training queue: {e}")

    return jsonify({'message': 'Data queued for training'})

# Note: /performance and /training-log endpoints are commented out due to missing functions
# @app.route('/performance', methods=['GET'])
# def performance():
#     metrics = get_performance_metrics()
#     return jsonify(metrics)

# @app.route('/training-log', methods=['GET'])
# def training_log():
#     log = get_training_log()
#     return jsonify(log)

# Placeholder for unimplemented endpoints from the summary (to be implemented if needed)
# @app.route('/', methods=['GET'])
# def serve_index():
#     return app.send_static_file('index.html')

# @app.route('/download-model', methods=['GET'])
# def download_model():
#     pass

# @app.route('/download-log', methods=['GET'])
# def download_log():
#     pass

# @app.route('/download-loss-history', methods=['GET'])
# def download_loss_history():
#     pass

# @app.route('/download-stage2-log', methods=['GET'])
# def download_stage2_log():
#     pass

# @app.route('/download-prediction-outcome-log', methods=['GET'])
# def download_prediction_outcome_log():
#     pass

# @app.route('/delete-log', methods=['DELETE'])
# def delete_log():
#     pass

# @app.route('/delete-loss-history', methods=['DELETE'])
# def delete_loss_history():
#     pass

# @app.route('/delete-stage2-log', methods=['DELETE'])
# def delete_stage2_log():
#     pass

# @app.route('/delete-prediction-outcome-log', methods=['DELETE'])
# def delete_prediction_outcome_log():
#     pass

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)

# Summary of Current Endpoints:
# /predict (POST): Receives prediction data (Last_30_Multipliers, etc.), predicts the next value using a hybrid CNN-LSTM model,
#                  assigns safety_label and RTP_window, returns a JSON with predicted_multiplier, confidence_score, safety_label, and RTP_window.
# /train (POST): Receives logging data (Skipped_or_Played, Your_Bet_Size, etc.), queues it for asynchronous training,
#                returns a JSON success message or error if data is invalid.
# /delete-model (DELETE): Deletes the model file (best_lstm_model.pth) if it exists, returns a JSON success or error message.
# /upload-model (POST): Uploads a new best_lstm_model.pth file, returns a JSON success message or error if invalid.
# (Other endpoints from the original summary are commented out and need implementation if required.)