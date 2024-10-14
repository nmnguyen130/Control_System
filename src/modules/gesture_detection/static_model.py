import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F

# 1. Dataset: Static Gesture Dataset từ file CSV
class StaticGestureDataset(Dataset):
    def __init__(self, csv_file, label_handler):
        data = pd.read_csv(csv_file)
        self.label_handler = label_handler

        # Normalize features and encode labels
        self.features = data.iloc[:, :-1].values
        self.labels = data['label'].apply(self.label_handler.get_static_value_by_gesture).values

        # Convert to tensors
        self.features = torch.tensor(self.features, dtype=torch.float32)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

# 2. Residual Block: Thêm kết nối tắt (skip connection)
class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features, in_features),
            nn.LayerNorm(in_features),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):
        return x + self.fc(x)  # Kết nối tắt (skip connection)

# 3. Embedding Model: Tạo vector embedding với residual blocks
class EmbeddingModel(nn.Module):
    def __init__(self):
        super(EmbeddingModel, self).__init__()
        self.fc1 = nn.Linear(63, 256)
        self.res_block1 = ResidualBlock(256)
        self.fc2 = nn.Linear(256, 128)
        self.res_block2 = ResidualBlock(128)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.res_block1(x)
        x = F.relu(self.fc2(x))
        x = self.res_block2(x)
        return x
    
# 3. Classification Model: Phân loại dựa trên vector embedding
class ClassificationModel(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationModel, self).__init__()
        self.fc1 = nn.Linear(128, 64)
        self.dropout = nn.Dropout(0.4)  # Tăng dropout để chống overfitting
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return F.log_softmax(self.fc2(x), dim=1)

# 5. Full Model: Kết hợp Embedding và Classification
class StaticGestureModel(nn.Module):
    def __init__(self, num_classes):
        super(StaticGestureModel, self).__init__()
        self.embedding = EmbeddingModel()
        self.classifier = ClassificationModel(num_classes)

    def forward(self, x):
        embeddings = self.embedding(x)
        return self.classifier(embeddings)