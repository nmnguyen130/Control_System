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

# 2. Dynamic Gesture Model LSTM
class DynamicGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(DynamicGestureModel, self).__init__()
        self.lstm = nn.LSTM(63, 256, num_layers=2, batch_first=True, dropout=0.3)
        self.bn = nn.BatchNorm1d(256)  # Batch Normalization sau LSTM
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x) #  lstm_out: (batch_size, seq_len, hidden_dim)

        # Global Max Pooling trên toàn bộ chuỗi
        pooled_out, _ = torch.max(lstm_out, dim=1)  # (batch_size, hidden_dim)
        pooled_out = self.bn(pooled_out)  # Batch Normalization

        return self.fc(pooled_out)