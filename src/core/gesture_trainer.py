import os
import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from src.modules.gesture_detection import *
from src.handlers.label_handler import LabelHandler

class GestureTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_loss = float('inf')

    def train(self, num_epochs=50, save_path=None):
        """Huấn luyện mô hình và lưu mô hình tốt nhất."""
        self.model.train()

        for epoch in range(num_epochs):
            total_loss = self._train_one_epoch(epoch, num_epochs)
            val_loss = self.evaluate()

            if self.scheduler:
                self.scheduler.step(val_loss)

            # Lưu lại mô hình tốt nhất dựa trên val_loss
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if save_path:
                    self.save_model(save_path)

    def _train_one_epoch(self, epoch, num_epochs):
        """Huấn luyện một epoch."""
        self.model.train()
        total_loss = 0.0
        progress = tqdm(self.train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for features, labels in progress:
            features, labels = features.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(features)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        print(f"Epoch [{epoch + 1}/{num_epochs}] - Training Loss: {avg_loss:.6f}")
        return avg_loss

    def evaluate(self):
        self.model.eval()
        total_loss, correct = 0.0, 0

        with torch.no_grad():
            for batch in self.val_loader:
                features, labels = self._unpack_batch(batch)
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self.model(features)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / len(self.val_loader.dataset)
        print(f"Validation - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")
        return avg_loss

    def save_model(self, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            torch.save(self.model.state_dict(), file_path)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")

    def _unpack_batch(self, batch):
        features, labels = batch
        return features, labels

def setup_static_model(label_handler):
    """Setup static gesture model, data loaders, and trainer."""
    static_csv_file = 'data/hand_gesture/static_data.csv'
    static_dataset = StaticGestureDataset(static_csv_file, label_handler=label_handler)
    
    train_data, val_data = train_test_split(static_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    static_num_classes = len(label_handler.static_labels.keys())
    static_model = StaticGestureModel(static_num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(static_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    return static_model, train_loader, val_loader, criterion, optimizer, scheduler

def setup_dynamic_model(label_handler):
    dynamic_csv_file = 'data/hand_gesture/dynamic_data.csv'
    dynamic_dataset = DynamicGestureDataset(dynamic_csv_file, label_handler=label_handler)
    
    train_data, val_data = train_test_split(dynamic_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    dynamic_num_classes = len(label_handler.dynamic_labels.keys())
    dynamic_model = DynamicGestureModel(dynamic_num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(dynamic_model.parameters(), lr=0.0005, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    return dynamic_model, train_loader, val_loader, criterion, optimizer, scheduler

def main():
    label_handler = LabelHandler('data/hand_gesture/static_labels.csv', 
                                 'data/hand_gesture/dynamic_labels.csv')
    
    # Setup and train static model
    static_trainer = GestureTrainer(*setup_static_model(label_handler))
    static_trainer.train(num_epochs=50, save_path='trained_data/static_gesture_model.pth')

    # Setup and train dynamic model
    dynamic_trainer = GestureTrainer(*setup_dynamic_model(label_handler))
    dynamic_trainer.train(num_epochs=50, save_path='trained_data/dynamic_gesture_model.pth')

if __name__ == "__main__":
    main()