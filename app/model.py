# app/model.py
import torch
import torch.nn as nn

class HybridCNNLSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=32, num_lstm_layers=1, max_seq_length=30, num_classes=5):
        super(HybridCNNLSTMModel, self).__init__()
        self.max_seq_length = max_seq_length
        
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
        
        self.lstm_input_size = 32  # Number of output channels from the last Conv1d
        self.lstm = nn.LSTM(input_size=self.lstm_input_size, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True)
        
        self.multiplier_head = nn.Linear(hidden_size, 1)
        self.rtp_head = nn.Linear(hidden_size, 1)
        self.classifier_head = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, channels=1, seq_length)
        cnn_out = self.cnn(x)  # Shape: (batch_size, out_channels=32, seq_length//4)
        
        # Transpose to (batch_size, seq_length//4, out_channels) for LSTM
        cnn_out = cnn_out.permute(0, 2, 1)  # Shape: (batch_size, seq_length//4, 32)
        
        # Pass through LSTM
        lstm_out, _ = self.lstm(cnn_out)  # Shape: (batch_size, seq_length//4, hidden_size)
        
        # Take the last timestep's output
        lstm_out = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)
        
        # Heads for different outputs
        predicted_multiplier = self.multiplier_head(lstm_out)  # Shape: (batch_size, 1)
        confidence_score = torch.sigmoid(self.rtp_head(lstm_out)) * 100  # Shape: (batch_size, 1)
        classifier_out = self.classifier_head(lstm_out)  # Shape: (batch_size, num_classes)
        
        return {
            'predicted_multiplier': predicted_multiplier,
            'confidence_score': confidence_score,
            'classifier_output': classifier_out
        }