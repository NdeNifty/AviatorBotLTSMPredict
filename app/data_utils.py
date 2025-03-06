import os
import json
import queue
from statistics import mean, stdev
import torch

# Define BASE_DIR based on persistent disk (update if mounted, e.g., '/data')
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Default to project dir
# If using a persistent disk mounted at /data, uncomment and set:
# BASE_DIR = '/data'

MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_lstm_model.pth')
TRAINING_LOG_PATH = os.path.join(BASE_DIR, 'model', 'training_log.json')
LOSS_HISTORY_PATH = os.path.join(BASE_DIR, 'model', 'loss_history.json')
STAGE2_LOG_PATH = os.path.join(BASE_DIR, 'model', 'stage2_log.json')
PREDICTION_OUTCOME_LOG_PATH = os.path.join(BASE_DIR, 'model', 'prediction_outcome_log.json')
TRAINING_QUEUE_PATH = os.path.join(BASE_DIR, 'model', 'training_queue.json')

# Ensure the model directory exists
os.makedirs(os.path.join(BASE_DIR, 'model'), exist_ok=True)

# Global variables and constants
save_interval = 10
training_queue = queue.Queue()

# Global variables
model = None
training_log = []
loss_history = []
data_buffer = queue.Queue(maxsize=1000)
data_min, data_max = 1.0, 100.0
last_sequence = None
min_seq_length = 10
max_seq_length = 30
stage2_log = []
prediction_outcome_log = []

def initialize_model():
    global model
    from .model import HybridCNNLSTMModel
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device in initialize_model: {device}")
    
    # Create the model instance
    try:
        model = HybridCNNLSTMModel(max_seq_length=max_seq_length).to(device)
        print(f"Model created successfully: {model}")
    except Exception as e:
        print(f"Error creating model: {e}")
        return None

    # Attempt to load from persistent disk
    try:
        state_dict = torch.load(MODEL_PATH, map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print("Loaded existing model from persistent disk with potential mismatches")
    except (FileNotFoundError, RuntimeError) as e:
        try:
            state_dict = torch.load('best_lstm_model.pth', map_location=device)
            model.load_state_dict(state_dict, strict=False)
            print("Loaded model from GitHub file with potential mismatches, saving to persistent disk")
            torch.save(model.state_dict(), MODEL_PATH)
        except (FileNotFoundError, RuntimeError) as e:
            print(f"Failed to load model due to {e}. Starting with a fresh model")
    
    # Verify and return model
    print(f"Model after initialization: {model}")
    return model