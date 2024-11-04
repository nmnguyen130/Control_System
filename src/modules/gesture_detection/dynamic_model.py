import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# 1. Dataset: Dynamic Gesture Dataset
class DynamicGestureDataset(Dataset):
    def __init__(self, csv_file, label_handler):
        data = pd.read_csv(csv_file)
        
        frames = data['frame'].values
        features = data.drop(columns=['frame', 'label']).values
        labels = data['label'].apply(label_handler.get_dynamic_value_by_gesture).values

        # Group sequences by label change or end of data
        self.samples, self.labels = self._group_sequences(features, frames, labels)

    def _group_sequences(self, features, frames, labels):
        samples, labels_list, temp_sample = [], [], []
        
        for i in range(len(features)):
            temp_sample.append(features[i])

            if i == len(features) - 1 or frames[i + 1] == 0:
                samples.append(torch.tensor(np.array(temp_sample), dtype=torch.float32))
                labels_list.append(labels[i])
                temp_sample = []  # Reset for next sample

        return samples, torch.tensor(labels_list, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# 2. Dynamic Gesture Model with Attention and Residual Connections
class DynamicGestureModel(nn.Module):
    def __init__(self, num_classes, input_dim=63, hidden_dim=128, num_layers=2, dropout=0.5):
        super(DynamicGestureModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Unidirectional LSTM
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        
        # Fully connected layers
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        
        # Dropout and activation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        
        # Take the output from the last time step
        last_time_step = lstm_out[:, -1, :]
        
        # Apply dropout and fully connected layers
        out = self.dropout(last_time_step)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out
