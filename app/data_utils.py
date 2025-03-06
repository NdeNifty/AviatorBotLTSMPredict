import os
import json
import queue  # Changed from collections.deque
from statistics import mean, stdev
import torch

# Define paths relative to the project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'best_lstm_model.pth')
TRAINING_LOG_PATH = os.path.join(BASE_DIR, 'model', 'training_log.json')
LOSS_HISTORY_PATH = os.path.join(BASE_DIR, 'model', 'loss_history.json')
STAGE2_LOG_PATH = os.path.join(BASE_DIR, 'model', 'stage2_log.json')
PREDICTION_OUTCOME_LOG_PATH = os.path.join(BASE_DIR, 'model', 'prediction_outcome_log.json')
TRAINING_QUEUE_PATH = os.path.join(BASE_DIR, 'model', 'training_queue.json')

# Ensure the model directory exists
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# Global variables and constants
save_interval = 10
training_queue = queue.Queue()  # Changed to queue.Queue

# Global variables
model = None
training_log = []
loss_history = []
data_buffer = queue.Queue(maxsize=1000)  # Changed to queue.Queue for consistency
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
    model = HybridCNNLSTMModel(max_seq_length=max_seq_length).to(device)
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