# app/training_utils.py
import torch
import torch.nn as nn  # Add this import
import threading
import queue
import os
import json
from datetime import datetime
from .data_utils import model, training_log, loss_history, data_buffer, data_min, data_max, min_seq_length, max_seq_length, MODEL_PATH, TRAINING_LOG_PATH, LOSS_HISTORY_PATH, STAGE2_LOG_PATH, stage2_log, PREDICTION_OUTCOME_LOG_PATH, prediction_outcome_log, save_interval

# Define these as None initially, and initialize them later
optimizer = None
criterion = None
scheduler = None
request_count = 0
training_queue = queue.Queue()
BATCH_SIZE = 10
VALIDATION_SPLIT = 0.2

def initialize_training_utils():
    global optimizer, criterion, scheduler
    if model is None:
        raise ValueError("Model must be initialized before training_utils")
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    criterion = nn.HuberLoss(delta=3.0)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5, verbose=True)

def load_or_init_training_queue():
    global training_queue
    TRAINING_QUEUE_PATH = '/model/training_queue.json'
    if os.path.exists(TRAINING_QUEUE_PATH):
        try:
            with open(TRAINING_QUEUE_PATH, 'r') as f:
                queued_data = json.load(f)
                for item in queued_data:
                    training_queue.put(item)
            print(f"Loaded training queue with {training_queue.qsize()} entries")
        except json.JSONDecodeError:
            print("Corrupted training queue, starting fresh with empty queue")
            training_queue = queue.Queue()
    else:
        print("No training queue found, starting fresh with empty queue")
        training_queue = queue.Queue()
    return training_queue

training_queue = load_or_init_training_queue()

def assign_safety_label(confidence, predicted_multiplier):
    if confidence > 70 and predicted_multiplier > 4:
        return "Safe"
    elif confidence > 70 and predicted_multiplier <= 4:
        return "Danger"
    else:
        return "Skip"

def assign_rtp_window(sequence):
    last_10 = sequence[-10:] if len(sequence) >= 10 else sequence
    return "Manipulated" if mean(last_10) < 2.0 else "Normal"

def train_batch(queued_data):
    global request_count
    if not queued_data:
        return
    
    sequences = []
    actual_multipliers = []
    for data in queued_data:
        logging_data = data.get('loggingData', {})
        sequence = logging_data.get('Last_30_Multipliers', [])
        actual_multiplier = logging_data.get('Actual_Multiplier', 1.0)
        if len(sequence) >= min_seq_length:
            seq_length = min(len(sequence), max_seq_length)
            sequences.append(sequence[-seq_length:])
            actual_multipliers.append(actual_multiplier)

    if not sequences:
        return

    seq_tensor = torch.stack([torch.FloatTensor([(x - data_min) / (data_max - data_min) for x in seq]).unsqueeze(0).unsqueeze(0) for seq in sequences])
    actual_tensor = torch.FloatTensor(actual_multipliers).unsqueeze(1)

    model.train()
    outputs = model(seq_tensor)
    predicted_multipliers = outputs['predicted_multiplier']
    confidence_scores = outputs['confidence_score']
    classifier_out = outputs['classifier_output']

    safety_labels = [assign_safety_label(confidence, pred) for confidence, pred in zip(confidence_scores.tolist(), predicted_multipliers.tolist())]
    rtp_windows = [assign_rtp_window(seq) for seq in sequences]
    classifier_labels = torch.tensor([0 if label == "Safe" else 1 if label == "Danger" else 2 for label in safety_labels])

    multiplier_loss = criterion(predicted_multipliers, actual_tensor)
    classifier_loss = nn.CrossEntropyLoss()(classifier_out, classifier_labels)
    total_loss = multiplier_loss + classifier_loss

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step(total_loss)

    loss_history.append(float(total_loss.item()))
    if len(loss_history) > 1000:
        loss_history = loss_history[-1000:]
    try:
        with open(LOSS_HISTORY_PATH, 'w') as f:
            json.dump(loss_history, f)
    except (IOError, json.JSONEncodeError) as e:
        print(f"Error saving loss history: {e}")

    for i, (seq, pred, act, conf, safe, rtp) in enumerate(zip(sequences, predicted_multipliers.tolist(), actual_multipliers, confidence_scores.tolist(), safety_labels, rtp_windows)):
        training_log.append({
            'sequence': seq,
            'predicted': float(pred),
            'actual': float(act),
            'prediction_data': {'confidence_score': float(conf), 'safety_label': safe, 'RTP_window': rtp}
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

    for data in queued_data:
        stage2_log.append(data.get('loggingData', {}))
    try:
        with open(STAGE2_LOG_PATH, 'w') as f:
            json.dump(stage2_log, f)
        print(f"Updated stage2 log with {len(stage2_log)} entries")
    except (IOError, json.JSONEncodeError) as e:
        print(f"Error saving stage2 log: {e}. Truncating log to last 1000 entries.")
        stage2_log = stage2_log[-1000:]
        with open(STAGE2_LOG_PATH, 'w') as f:
            json.dump(stage2_log, f)

    for i, (seq, pred, act, conf, safe, rtp) in enumerate(zip(sequences, predicted_multipliers.tolist(), actual_multipliers, confidence_scores.tolist(), safety_labels, rtp_windows)):
        prediction_outcome_log.append({
            'predicted_multiplier': float(pred),
            'confidence_score': float(conf),
            'safety_label': safe,
            'RTP_window': rtp,
            'actual_multiplier': float(act),
            'sequence': seq
        })
    try:
        with open(PREDICTION_OUTCOME_LOG_PATH, 'w') as f:
            json.dump(prediction_outcome_log, f)
        print(f"Updated prediction outcome log with {len(prediction_outcome_log)} entries")
    except (IOError, json.JSONEncodeError) as e:
        print(f"Error saving prediction outcome log: {e}. Truncating log to last 1000 entries.")
        prediction_outcome_log = prediction_outcome_log[-1000:]
        with open(PREDICTION_OUTCOME_LOG_PATH, 'w') as f:
            json.dump(prediction_outcome_log, f)

    if request_count >= save_interval:
        torch.save(model.state_dict(), MODEL_PATH)
        print(f'Saved model to persistent disk after {request_count} requests')
        request_count = 0

    print(f"Trained batch - Batch Size: {len(sequences)}, Loss: {total_loss.item():.4f}")

def training_loop():
    global request_count
    while True:
        queued_data = []
        while not training_queue.empty() and len(queued_data) < BATCH_SIZE:
            queued_data.append(training_queue.get())
        
        if queued_data:
            train_batch(queued_data)
            for _ in queued_data:
                training_queue.task_done()
        
        if training_log and len(training_log) > 10:
            val_size = int(len(training_log) * VALIDATION_SPLIT)
            train_data = training_log[:-val_size]
            val_data = training_log[-val_size:]
            val_loss = calculate_validation_loss(val_data)
            print(f"Validation Loss: {val_loss:.4f}")
            if val_loss > previous_val_loss:  # Needs tuning
                print("Validation loss increased, considering early stopping")
                break
        
        threading.Event().wait(60)

def calculate_validation_loss(val_data):
    sequences = [entry['sequence'] for entry in val_data]
    actuals = [entry['actual'] for entry in val_data]
    seq_tensor = torch.stack([torch.FloatTensor([(x - data_min) / (data_max - data_min) for x in seq]).unsqueeze(0).unsqueeze(0) for seq in sequences])
    actual_tensor = torch.FloatTensor(actuals).unsqueeze(1)
    
    with torch.no_grad():
        model.eval()
        outputs = model(seq_tensor)
        predicted_multipliers = outputs['predicted_multiplier']
        return criterion(predicted_multipliers, actual_tensor).item()

# Start training thread
training_thread = threading.Thread(target=training_loop, daemon=True)
training_thread.start()