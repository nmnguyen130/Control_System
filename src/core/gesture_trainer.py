import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from src.modules.gesture_detection import *
from src.handlers.label_handler import LabelHandler

def collate_fn(batch):
        """Collate function for handling variable-length sequences in dynamic gesture dataset."""
        # Sort batch by sequence length in descending order
        batch.sort(key=lambda x: x[2], reverse=True)
        sequences, labels, lengths = zip(*batch)
        
        # Pad sequences to the length of the longest sequence in the batch
        padded_sequences = nn.utils.rnn.pad_sequence(sequences, batch_first=True)
        return padded_sequences, torch.tensor(labels, dtype=torch.long), lengths

class EarlyStopping:
    def __init__(self, patience=5):
        self.patience = patience
        self.counter = 0
        self.best_loss = np.inf

    def step(self, val_loss):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

class GestureTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs=50):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for batch in self.train_loader:
                features, labels, lengths = self._unpack_batch(batch)
                features, labels = features.to(self.device), labels.to(self.device)

                self.optimizer.zero_grad()
                outputs = self._forward_model(features, lengths)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch + 1}/{num_epochs}], Training Loss: {avg_loss:.6f}")

    def evaluate(self):
        self.model.eval()
        total_loss, correct = 0.0, 0
        with torch.no_grad():
            for batch in self.val_loader:
                features, labels, lengths = self._unpack_batch(batch)
                features, labels = features.to(self.device), labels.to(self.device)

                outputs = self._forward_model(features, lengths)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                correct += (predicted == labels).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / len(self.val_loader.dataset)
        print(f"Validation - Loss: {avg_loss:.6f}, Accuracy: {accuracy:.2f}%")

    def save_model(self, file_path):
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def _unpack_batch(self, batch):
        if isinstance(self.model, DynamicGestureModel):
            return batch
        else:
            features, labels = batch
            return features, labels, None

    def _is_dynamic_model(self):
        return isinstance(self.model, DynamicGestureModel)
    
    def _forward_model(self, features, lengths):
        """Forward pass through the model based on its type (static/dynamic)."""
        return self.model(features, lengths) if lengths else self.model(features)

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
    optimizer = torch.optim.Adam(static_model.parameters(), lr=0.001)

    return static_model, train_loader, val_loader, criterion, optimizer

def setup_dynamic_model(label_handler):
    """Setup dynamic gesture model, data loaders, and trainer."""
    dynamic_csv_file = 'data/hand_gesture/dynamic_data.csv'
    dynamic_dataset = DynamicGestureDataset(dynamic_csv_file, label_handler=label_handler)
    
    train_data, val_data = train_test_split(dynamic_dataset, test_size=0.2, random_state=42)
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False, collate_fn=collate_fn)

    dynamic_num_classes = len(label_handler.dynamic_labels.keys())
    dynamic_model = DynamicGestureModel(dynamic_num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(dynamic_model.parameters(), lr=0.001)

    return dynamic_model, train_loader, val_loader, criterion, optimizer

def main():
    label_handler = LabelHandler('data/hand_gesture/static_labels.csv', 
                                 'data/hand_gesture/dynamic_labels.csv')
    
    # Setup and train static model
    # static_trainer = GestureTrainer(*setup_static_model(label_handler))
    # static_trainer.train(num_epochs=50)
    # static_trainer.evaluate()
    # static_trainer.save_model('trained_data/static_gesture_model.pth')

    # Setup and train dynamic model
    dynamic_trainer = GestureTrainer(*setup_dynamic_model(label_handler))
    dynamic_trainer.train(num_epochs=50)
    dynamic_trainer.evaluate()
    dynamic_trainer.save_model('trained_data/dynamic_gesture_model.pth')

if __name__ == "__main__":
    main()