import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# 1. Dataset: Dynamic Gesture Dataset
class DynamicGestureDataset(Dataset):
    def __init__(self, csv_file, label_handler):
        data = pd.read_csv(csv_file)
        self.label_handler = label_handler
        
        frames = data['frame'].values
        features = data.drop(columns=['frame', 'label']).values
        labels = data['label'].apply(self.label_handler.get_dynamic_value_by_gesture).values

        # Group sequences by label change or end of data
        self.samples, self.labels, self.lengths = [], [], []

        temp_sample = []
        for i in range(len(features)):
            temp_sample.append(features[i])

            if i == len(features) - 1 or frames[i + 1] == 0:
                self.samples.append(torch.tensor(np.array(temp_sample), dtype=torch.float32))
                self.labels.append(labels[i])
                self.lengths.append(len(temp_sample))
                temp_sample = []  # Reset for next sample

        # Convert lists of labels into a single tensor
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return the sequence, label, and its length
        return self.samples[idx], self.labels[idx], self.lengths[idx]
    
# 2. Attention Mechanism
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)

    def forward(self, lstm_output):
        # lstm_output: (batch_size, seq_len, 2 * hidden_size)
        attn_weights = torch.softmax(self.attention(lstm_output), dim=1)  # (batch_size, seq_len, 1)
        context = torch.sum(attn_weights * lstm_output, dim=1)  # Weighted sum
        return context

# 3. Dynamic Gesture Model với Bi-directional LSTM và Attention
class DynamicGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(DynamicGestureModel, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=2, 
                            batch_first=True, bidirectional=True)
        self.attention = Attention(128 * 2)  # Bi-directional LSTM có hidden size gấp đôi
        self.fc = nn.Linear(128 * 2, num_classes)

    def forward(self, x, lengths):
        # Pack và xử lý chuỗi có độ dài khác nhau
        packed_input = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_output, (h_n, _) = self.lstm(packed_input)
        lstm_output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Sử dụng Attention để tạo context vector
        context = self.attention(lstm_output)
        return self.fc(context)