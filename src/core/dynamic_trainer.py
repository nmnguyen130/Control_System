import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from src.modules.gesture_detection.dynamic_model import DynamicGestureDataset, DynamicGestureModel

class DynamicGestureTrainer:
    def __init__(self, csv_file, model_class, batch_size=32, epochs=50, learning_rate=0.001):
        self.csv_file = csv_file
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate

        self.model = model_class(num_classes=self._get_num_classes())  # Khởi tạo mô hình từ lớp truyền vào
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.data_loader = self._prepare_data_loader()

    def _prepare_data_loader(self):
        dataset = DynamicGestureDataset(self.csv_file)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
    
    def _get_num_classes(self):
        dataset = DynamicGestureDataset(self.csv_file)
        return len(dataset.label_encoder.classes_)

    def train(self):
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0
            for i, (inputs, labels) in enumerate(self.data_loader):
                inputs = inputs.unsqueeze(1)  # Kích thước: (batch_size, sequence_length=1, num_features)
                self.optimizer.zero_grad()  # Reset gradients
                outputs = self.model(inputs)  # Forward pass
                loss = self.criterion(outputs, labels)  # Calculate loss
                loss.backward()  # Backward pass
                self.optimizer.step()  # Update weights
                total_loss += loss.item()  # Accumulate loss

            # Print training information
            avg_loss = total_loss / len(self.data_loader)
            print(f'Epoch [{epoch+1}/{self.epochs}], Loss: {avg_loss:.4f}')

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model đã được lưu tại {file_path}!")

if __name__ == "__main__":
    # Huấn luyện mô hình cử chỉ tĩnh
    static_csv_file = 'data/hand_gesture/data_dynamic.csv'
    static_trainer = DynamicGestureTrainer(static_csv_file, DynamicGestureModel)
    static_trainer.train()
    static_trainer.save_model('trained_data/dynamic_gesture_model.pth')
