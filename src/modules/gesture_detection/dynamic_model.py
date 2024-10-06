import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class DynamicGestureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Chuẩn hóa dữ liệu
        self.scaler = StandardScaler()
        self.data.iloc[:, 1:-1] = self.scaler.fit_transform(self.data.iloc[:, 1:-1])

        # Mã hóa nhãn
        self.label_encoder = LabelEncoder()
        self.data['label'] = self.label_encoder.fit_transform(self.data['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx, 1:-1].values.astype('float32')  # Tất cả trừ cột frame và nhãn
        y = self.data.iloc[idx, -1]  # Cột nhãn
        return x, y

class DynamicGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(DynamicGestureModel, self).__init__()
        self.lstm = nn.LSTM(input_size=63, hidden_size=128, num_layers=2, batch_first=True)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])  # Chỉ lấy đầu ra cuối cùng
        return x