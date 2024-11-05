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

# 2. LSTM Block
class LSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super(LSTMBlock, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return lstm_out

# 3. Attention Block
class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionBlock, self).__init__()
        self.attention_fc = nn.Linear(hidden_dim, 1)
        
    def forward(self, lstm_out):
        # Compute attention weights
        attention_weights = torch.softmax(self.attention_fc(lstm_out), dim=1)
        
        # Apply attention weights to LSTM outputs
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        return context_vector
    
# 4. Fully Connected Block
class FullyConnectedBlock(nn.Module):
    def __init__(self, hidden_dim, num_classes, dropout=0.5):
        super(FullyConnectedBlock, self).__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, num_classes)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 2. Dynamic Gesture Model with Attention and Residual Connections
class DynamicGestureModel(nn.Module):
    def __init__(self, num_classes, input_dim=63, hidden_dim=128, num_layers=2, dropout=0.5):
        super(DynamicGestureModel, self).__init__()
        self.lstm = LSTMBlock(input_dim, hidden_dim, num_layers, dropout)
        self.attention = AttentionBlock(hidden_dim)
        self.fc = FullyConnectedBlock(hidden_dim, num_classes, dropout)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, 63)
        lstm_out = self.lstm(x)
        attention_out = self.attention(lstm_out)
        out = self.fc(attention_out)
        return out
