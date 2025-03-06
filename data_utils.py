# app/data_utils.py
import os
import json
from collections import deque
from statistics import mean, stdev

# Define paths on persistent disk
MODEL_PATH = '/model/best_lstm_model.pth'
TRAINING_LOG_PATH = '/model/training_log.json'
LOSS_HISTORY_PATH = '/model/loss_history.json'
STAGE2_LOG_PATH = '/model/stage2_log.json'
PREDICTION_OUTCOME_LOG_PATH = '/model/prediction_outcome_log.json'

# Global variables
model = None
training_log = []
loss_history = []
data_buffer = deque(maxlen=1000)
data_min, data_max = 1.0, 100.0
last_sequence = None
min_seq_length = 10
max_seq_length = 30

def initialize_model():
    global model
    from .model import HybridCNNLSTMModel
    model = HybridCNNLSTMModel(max_seq_length=max_seq_length)
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

def load_or_init_stage2_log():
    global stage2_log
    stage2_log = []
    if os.path.exists(STAGE2_LOG_PATH):
        try:
            with open(STAGE2_LOG_PATH, 'r') as f:
                stage2_log = json.load(f)
            print(f"Loaded stage2 log with {len(stage2_log)} entries")
            if len(stage2_log) > 1000:
                stage2_log = stage2_log[-1000:]
        except json.JSONDecodeError:
            print("Corrupted stage2 log, starting fresh with empty list")
            stage2_log = []
    return stage2_log

def load_or_init_prediction_outcome_log():
    global prediction_outcome_log
    prediction_outcome_log = []
    if os.path.exists(PREDICTION_OUTCOME_LOG_PATH):
        try:
            with open(PREDICTION_OUTCOME_LOG_PATH, 'r') as f:
                prediction_outcome_log = json.load(f)
            print(f"Loaded prediction outcome log with {len(prediction_outcome_log)} entries")
            if len(prediction_outcome_log) > 1000:
                prediction_outcome_log = prediction_outcome_log[-1000:]
        except json.JSONDecodeError:
            print("Corrupted prediction outcome log, starting fresh with empty list")
            prediction_outcome_log = []
    return prediction_outcome_log

def calculate_performance(log):
    if not log:
        return {
            "mae": 0.0,
            "within_one_percent": 0.0,
            "high_value_count": 0,
            "low_value_count": 0,
            "predicted_actual": []
        }
    
    mae = sum(abs(entry["predicted"] - entry["actual"]) for entry in log) / len(log)
    within_one = sum(1 for entry in log if abs(entry["predicted"] - entry["actual"]) <= 1.0) / len(log) * 100
    high_value_count = sum(1 for entry in log if entry["actual"] > 100.0)
    low_value_count = sum(1 for entry in log if entry["actual"] <= 100.0)
    
    predicted_actual = [(entry["predicted"], entry["actual"]) for entry in log[-250:]]
    
    return {
        "mae": float(mae),
        "within_one_percent": float(within_one),
        "high_value_count": high_value_count,
        "low_value_count": low_value_count,
        "predicted_actual": predicted_actual
    }

# Initialize additional logs
stage2_log = load_or_init_stage2_log()
prediction_outcome_log = load_or_init_prediction_outcome_log()