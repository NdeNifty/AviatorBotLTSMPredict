import torch
import torch.nn as nn

class HybridCNNLSTMModel(nn.Module):
    def __init__(self, max_seq_length):
        super(HybridCNNLSTMModel, self).__init__()
        self.max_seq_length = max_seq_length
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.lstm = nn.LSTM(input_size=64, hidden_size=128, batch_first=True)
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 1)  # Predicted multiplier
        self.fc_classifier = nn.Linear(64, 3)  # For safety label classification
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: (batch, 1, seq_length)
        x = self.relu(self.conv1(x))  # (batch, 64, seq_length)
        x = x.permute(0, 2, 1)  # (batch, seq_length, 64)
        lstm_out, _ = self.lstm(x)  # (batch, seq_length, 128)
        lstm_out = lstm_out[:, -1, :]  # Take the last time step (batch, 128)
        dense = self.relu(self.fc1(lstm_out))  # (batch, 64)
        predicted_multiplier = self.fc2(dense)  # (batch, 1)
        confidence_score = self.sigmoid(predicted_multiplier)  # (batch, 1)
        classifier_out = self.fc_classifier(dense)  # (batch, 3)
        return {
            'predicted_multiplier': predicted_multiplier,
            'confidence_score': confidence_score,
            'classifier_output': classifier_out
        }