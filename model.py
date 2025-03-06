# app/model.py
import torch
import torch.nn as nn

class HybridCNNLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_lstm_layers=1, max_seq_length=30, num_classes=5):
        super(HybridCNNLSTMModel, self).__init__()
        self.max_seq_length = max_seq_length
        
        # CNN Head for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2)
        )
        
        # Reshape for LSTM (after CNN)
        self.lstm_input_size = 32 * (max_seq_length // 4)  # Adjusted for pooling
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True)
        
        # Multi-head outputs
        self.multiplier_head = nn.Linear(hidden_size, 1)  # Predict multiplier
        self.rtp_head = nn.Linear(hidden_size, 1)       # Confidence score (0-100)
        self.classifier_head = nn.Linear(hidden_size, num_classes)  # Safety label (3) + RTP window (2)

    def forward(self, x):
        # x shape: (batch_size, 1, max_seq_length)
        cnn_out = self.cnn(x)  # Shape: (batch_size, 32, seq_length//4)
        cnn_out = cnn_out.permute(0, 2, 1)  # Reshape for LSTM: (batch_size, seq_length//4, 32)
        
        lstm_out, _ = self.lstm(cnn_out)  # Shape: (batch_size, seq_length//4, hidden_size)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step
        
        multiplier = self.multiplier_head(lstm_out)  # Predict multiplier
        confidence = torch.sigmoid(self.rtp_head(lstm_out)) * 100  # Confidence score (0-100)
        classifier_out = self.classifier_head(lstm_out)  # Raw outputs for classification
        
        return {
            'predicted_multiplier': multiplier.squeeze(),
            'confidence_score': confidence.squeeze(),
            'classifier_output': classifier_out
        }