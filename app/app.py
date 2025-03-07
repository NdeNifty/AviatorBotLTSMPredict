from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from .data_utils import initialize_model, data_buffer, data_min, data_max, last_sequence, max_seq_length, training_queue, TRAINING_QUEUE_PATH, model
from .training_utils import assign_safety_label, assign_rtp_window, initialize_training_utils, training_loop
import os
import json
from datetime import datetime
import time
import threading

app = Flask(__name__)
CORS(app)

# Initialize device for GPU support
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize model with GPU support
print("Calling initialize_model()")
initialized_model = initialize_model()

# Debug model state before proceeding
print(f"Model after initialize_model: {initialized_model}")
if initialized_model is None:
    raise ValueError("Model initialization failed")

# Assign the initialized model to the global model variable
global model
model = initialized_model

# Initialize training utilities after model is confirmed
initialize_training_utils(model)

# Start the training thread with the initialized model
training_thread = threading.Thread(target=training_loop, args=(model,), daemon=True)
training_thread.start()

# Route to predict the next multiplier
@app.route('/predict', methods=['POST'])
def predict():
    global model, data_min, data_max
    from .data_utils import data_buffer, data_min, data_max, last_sequence, max_seq_length
    
    prediction_data = request.json.get('predictionData', {})
    sequence = prediction_data.get('Last_30_Multipliers', [])
    time_gaps = prediction_data.get('Time_Gaps', None)

    if not sequence:
        return jsonify({'error': 'No sequence provided'}), 400

    if time_gaps is not None:
        print(f"Received Time_Gaps: {time_gaps}")
    else:
        print("No Time_Gaps provided")

    seq_length = min(len(sequence), max_seq_length)
    padded_sequence = sequence[-seq_length:]
    while len(padded_sequence) < 10:
        padded_sequence.insert(0, 1.0)
    
    print(f"Padded sequence: {padded_sequence}")
    data_buffer.extend(padded_sequence)
    print(f"Data buffer after extend (length {len(data_buffer)}): {list(data_buffer)}")
    data_min = min(data_min, min(padded_sequence))
    data_max = min(max(data_max, max(padded_sequence)), 300.0)
    print(f"Updated data_min: {data_min}, data_max: {data_max}")
    seq_normalized = [(x - data_min) / (data_max - data_min) for x in padded_sequence]
    seq_tensor = torch.FloatTensor(seq_normalized).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        model.eval()
        model.to(device)
        outputs = model(seq_tensor)
        predicted_multiplier = outputs['predicted_multiplier'].item()
        confidence_score = outputs['confidence_score'].item()
        classifier_out = outputs['classifier_output']
        
        safety_label = assign_safety_label(confidence_score * 100, predicted_multiplier)
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
            model = HybridCNNLSTMModel(max_seq_length=max_seq_length).to(device)
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

    if 'Actual_Multiplier' not in data:
        data['Actual_Multiplier'] = data.get('Multiplier_Outcome', 1.0)

    training_queue.put({'loggingData': data, 'timestamp': datetime.utcnow().isoformat()})
    try:
        with open(TRAINING_QUEUE_PATH, 'w') as f:
            json.dump(list(training_queue.queue), f)
        print(f"Added to training queue, total: {training_queue.qsize()}")
    except (IOError, json.JSONEncodeError) as e:
        print(f"Error saving training queue: {e}")

    return jsonify({'message': 'Data queued for training'})

# Placeholder for unimplemented endpoints from the summary
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