import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

class StaticGestureDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Chuẩn hóa dữ liệu
        self.scaler = StandardScaler()
        self.data.iloc[:, :-1] = self.scaler.fit_transform(self.data.iloc[:, :-1])

        # Mã hóa nhãn
        self.label_encoder = LabelEncoder()
        self.data['label'] = self.label_encoder.fit_transform(self.data['label'])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = self.data.iloc[idx, :-1].values.astype('float32')  # Tất cả trừ cột nhãn
        y = self.data.iloc[idx, -1]  # Cột nhãn
        return x, y

# Định nghĩa mô hình
class StaticGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(StaticGestureModel, self).__init__()
        self.fc1 = nn.Linear(63, 128)  # 63 cho 21 điểm (x, y, z)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)  # nhãn cử chỉ và không nhãn

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x