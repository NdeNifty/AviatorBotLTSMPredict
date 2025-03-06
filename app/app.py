# app/app.py
import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from .data_utils import initialize_model, load_or_init_training_log, load_or_init_loss_history, calculate_performance

# Initialize model and data first
initialize_model()

# Now import training_utils and initialize its components
from .training_utils import training_queue, model, assign_safety_label, assign_rtp_window, initialize_training_utils
initialize_training_utils()

app = Flask(__name__, static_folder='static')
CORS(app, resources={r"/performance": {"origins": "https://aviatorbotltsmpredict.onrender.com"}, r"/training-log": {"origins": "https://aviatorbotltsmpredict.onrender.com"}})

training_log = load_or_init_training_log()
loss_history = load_or_init_loss_history()

# Rest of the file remains the same...
@app.route('/predict', methods=['POST'])
def predict():
    global model
    from .data_utils import data_buffer, data_min, data_max, last_sequence, max_seq_length
    
    sequence = request.json.get('predictionData', {}).get('Last_30_Multipliers', [])
    if not sequence or len(sequence) < 10:
        return jsonify({'error': 'Sequence must be at least 10 numbers'}), 400

    seq_length = min(len(sequence), max_seq_length)
    sequence = sequence[-seq_length:]
    
    data_buffer.extend(sequence)
    data_min = min(data_min, min(sequence))
    data_max = min(max(data_max, max(sequence)), 300.0)

    seq_normalized = [(x - data_min) / (data_max - data_min) for x in sequence]
    seq_tensor = torch.FloatTensor(seq_normalized).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        model.eval()
        outputs = model(seq_tensor)
        predicted_multiplier = outputs['predicted_multiplier'].item()
        confidence_score = outputs['confidence_score'].item()
        classifier_out = outputs['classifier_output']
        
        safety_label = assign_safety_label(confidence_score, predicted_multiplier)
        rtp_window = assign_rtp_window(sequence)
        
        response = {
            'predicted_multiplier': float(predicted_multiplier),
            'confidence_score': float(confidence_score),
            'safety_label': safety_label,
            'RTP_window': rtp_window
        }
        return jsonify(response)


if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

@app.route('/train', methods=['POST'])
def train():
    global training_queue
    data = request.json.get('loggingData', {})
    if not data:
        return jsonify({'error': 'No logging data provided'}), 400

    training_queue.put({'loggingData': data, 'timestamp': datetime.utcnow().isoformat()})
    try:
        with open('/model/training_queue.json', 'w') as f:
            json.dump(list(training_queue.queue), f)
        print(f"Added to training queue, total: {training_queue.qsize()}")
    except (IOError, json.JSONEncodeError) as e:
        print(f"Error saving training queue: {e}")

    return jsonify({'message': 'Data queued for training'})

@app.route('/performance', methods=['GET'])
def get_performance():
    from .data_utils import calculate_performance, training_log, loss_history
    performance = calculate_performance(training_log)
    response = {
        "performance": performance,
        "learning_curve": loss_history
    }
    return jsonify(response)

@app.route('/training-log', methods=['GET'])
def get_training_log():
    from .data_utils import TRAINING_LOG_PATH
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
    from .data_utils import MODEL_PATH
    if os.path.exists(MODEL_PATH):
        return send_file(MODEL_PATH, as_attachment=True, download_name='best_lstm_model.pth')
    else:
        return jsonify({'error': 'Model file not found. Train the model first.'}), 404

@app.route('/download-log', methods=['GET'])
def download_log():
    from .data_utils import TRAINING_LOG_PATH
    if os.path.exists(TRAINING_LOG_PATH):
        return send_file(TRAINING_LOG_PATH, as_attachment=True, download_name='training_log.json')
    else:
        return jsonify({'error': 'Training log not found. Train the model first.'}), 404

@app.route('/download-loss-history', methods=['GET'])
def download_loss_history():
    from .data_utils import LOSS_HISTORY_PATH
    if os.path.exists(LOSS_HISTORY_PATH):
        return send_file(LOSS_HISTORY_PATH, as_attachment=True, download_name='loss_history.json')
    else:
        return jsonify({'error': 'Loss history not found. Train the model first.'}), 404

@app.route('/download-stage2-log', methods=['GET'])
def download_stage2_log():
    from .data_utils import STAGE2_LOG_PATH
    if os.path.exists(STAGE2_LOG_PATH):
        return send_file(STAGE2_LOG_PATH, as_attachment=True, download_name='stage2_log.json')
    else:
        return jsonify({'error': 'Stage 2 log not found.'}), 404

@app.route('/download-prediction-outcome-log', methods=['GET'])
def download_prediction_outcome_log():
    from .data_utils import PREDICTION_OUTCOME_LOG_PATH
    if os.path.exists(PREDICTION_OUTCOME_LOG_PATH):
        return send_file(PREDICTION_OUTCOME_LOG_PATH, as_attachment=True, download_name='prediction_outcome_log.json')
    else:
        return jsonify({'error': 'Prediction outcome log not found.'}), 404

@app.route('/upload-model', methods=['POST'])
def upload_model():
    from .data_utils import MODEL_PATH, model
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
    from .data_utils import training_log, TRAINING_LOG_PATH
    if os.path.exists(TRAINING_LOG_PATH):
        os.remove(TRAINING_LOG_PATH)
        training_log = []
        print("Training log deleted successfully")
        return jsonify({'message': 'Training log deleted successfully'}), 200
    else:
        return jsonify({'error': 'Training log not found'}), 404

@app.route('/delete-loss-history', methods=['DELETE'])
def delete_loss_history():
    from .data_utils import loss_history, LOSS_HISTORY_PATH
    if os.path.exists(LOSS_HISTORY_PATH):
        os.remove(LOSS_HISTORY_PATH)
        loss_history = []
        print("Loss history deleted successfully")
        return jsonify({'message': 'Loss history deleted successfully'}), 200
    else:
        return jsonify({'error': 'Loss history not found'}), 404

@app.route('/delete-stage2-log', methods=['DELETE'])
def delete_stage2_log():
    from .data_utils import stage2_log, STAGE2_LOG_PATH
    if os.path.exists(STAGE2_LOG_PATH):
        os.remove(STAGE2_LOG_PATH)
        stage2_log = []
        print("Stage 2 log deleted successfully")
        return jsonify({'message': 'Stage 2 log deleted successfully'}), 200
    else:
        return jsonify({'error': 'Stage 2 log not found'}), 404

@app.route('/delete-prediction-outcome-log', methods=['DELETE'])
def delete_prediction_outcome_log():
    from .data_utils import prediction_outcome_log, PREDICTION_OUTCOME_LOG_PATH
    if os.path.exists(PREDICTION_OUTCOME_LOG_PATH):
        os.remove(PREDICTION_OUTCOME_LOG_PATH)
        prediction_outcome_log = []
        print("Prediction outcome log deleted successfully")
        return jsonify({'message': 'Prediction outcome log deleted successfully'}), 200
    else:
        return jsonify({'error': 'Prediction outcome log not found'}), 404

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)

# Summary of Endpoints:
# / (GET): Serves the static index.html for visualization, loading charts for performance metrics, learning curve, and predicted vs. actual payouts.
# /predict (POST): Receives prediction data (Last_30_Multipliers, etc.), predicts the next value using a hybrid CNN-LSTM model, assigns safety_label and RTP_window,
#                  returns a JSON with predicted_multiplier, confidence_score, safety_label, and RTP_window. Saves model every 10 training steps to persistent disk
#                  and logs predicted/actual values to training_log.json and loss to loss_history.json.
# /train (POST): Receives logging data (Skipped_or_Played, Your_Bet_Size, etc.), queues it for asynchronous training,
#                returns a JSON success message or error if data is invalid.
# /performance (GET): Returns JSON with performance metrics (MAE, % within ±1.0, high/low value counts) and learning curve (loss history).
# /training-log (GET): Returns JSON of the training_log.json file from persistent disk, containing sequence, predicted, and actual values for visualization.
# /download-model (GET): Downloads the trained model file (best_lstm_model.pth) from the persistent disk if it exists,
#                       returns a JSON error if not found.
# /download-log (GET): Downloads the training log file (training_log.json) containing sequence, predicted, and actual values,
#                     returns a JSON error if not found.
# /download-loss-history (GET): Downloads the loss history file (loss_history.json) containing loss values over time,
#                              returns a JSON error if not found.
# /download-stage2-log (GET): Downloads the stage2 log file (stage2_log.json) containing Stage 2 data,
#                            returns a JSON error if not found.
# /download-prediction-outcome-log (GET): Downloads the prediction outcome log file (prediction_outcome_log.json) containing prediction outcomes,
#                                        returns a JSON error if not found.
# /upload-model (POST): Uploads a new best_lstm_model.pth file to overwrite any existing model on the persistent disk,
#                      returns a JSON success message or error if the file is invalid or upload fails.
# /delete-log (DELETE): Deletes the training log file (training_log.json) from the persistent disk if it exists, resets the in-memory training_log.
# /delete-loss-history (DELETE): Deletes the loss history file (loss_history.json) from the persistent disk if it exists, resets the in-memory loss_history.
# /delete-stage2-log (DELETE): Deletes the stage2 log file (stage2_log.json) from the persistent disk if it exists, resets the in-memory stage2_log.
# /delete-prediction-outcome-log (DELETE): Deletes the prediction outcome log file (prediction_outcome_log.json) from the persistent disk if it exists,
#                                         resets the in-memory prediction_outcome_log.
# Hope this helps!



