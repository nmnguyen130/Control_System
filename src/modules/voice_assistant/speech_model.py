import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class SpeechRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(SpeechRecognitionModel, self).__init__()
        
        # Convolutional layers for feature extraction
        self.conv1 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(256, 512, kernel_size=3, stride=1, padding=1)
        
        # Batch normalization layers
        self.bn1 = nn.BatchNorm1d(256)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(512)
        
        # Bidirectional LSTM layers
        self.lstm = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Apply CNN layers
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        
        # Prepare for LSTM (batch_size, sequence_length, features)
        x = x.transpose(1, 2)
        
        # Apply LSTM
        x, _ = self.lstm(x)
        
        # Take the output from the last time step
        x = x[:, -1, :]
        
        # Apply fully connected layers
        x = F.relu(self.fc1(self.dropout(x)))
        x = self.fc2(self.dropout(x))
        
        return x
