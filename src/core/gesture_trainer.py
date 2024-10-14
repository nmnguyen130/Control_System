import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

class GestureTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self, num_epochs=50):
        """Train the model for a specified number of epochs."""
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
        """Evaluate the model on the validation set."""
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
        """Save the trained model to the specified file path."""
        torch.save(self.model.state_dict(), file_path)
        print(f"Model saved to {file_path}")

    def _unpack_batch(self, batch):
        """Helper function to unpack batch data for static or dynamic models."""
        if self._is_dynamic_model():
            features, labels, lengths = batch
        else:
            features, labels = batch
            lengths = None
        return features, labels, lengths

    def _is_dynamic_model(self):
        """Check if the current model is a dynamic gesture model."""
        return isinstance(self.model, DynamicGestureModel)
    
    def _forward_model(self, features, lengths):
        """Forward pass through the model based on its type (static/dynamic)."""
        return self.model(features, lengths) if self._is_dynamic_model() else self.model(features)
    
if __name__ == "__main__":
    static_label_csv_path = 'data/hand_gesture/static_labels.csv'
    dynamic_label_csv_path = 'data/hand_gesture/dynamic_labels.csv'
    label_handler = LabelHandler(static_label_csv_path, dynamic_label_csv_path)

    # Example usage for static gesture recognition
    static_csv_file = 'data/hand_gesture/static_data.csv'
    static_dataset = StaticGestureDataset(static_csv_file, label_handler=label_handler)
    static_data_loader = DataLoader(static_dataset, batch_size=32, shuffle=True)

    # Model configuration
    static_num_classes = len(torch.unique(static_dataset.labels))

    static_model = StaticGestureModel(static_num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(static_model.parameters(), lr=0.001)

    # Train and evaluate static model
    trainer = GestureTrainer(static_model, static_data_loader, static_data_loader, criterion, optimizer)
    trainer.train(num_epochs=50)
    trainer.evaluate()

    # Save static model
    trainer.save_model('trained_data/static_gesture_model.pth')

    # Example usage for dynamic gesture recognition
    dynamic_csv_file = 'data/hand_gesture/dynamic_data.csv'  # Replace with actual path
    dynamic_dataset = DynamicGestureDataset(dynamic_csv_file, label_handler=label_handler)
    dynamic_data_loader = DataLoader(dynamic_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    # Model configuration for dynamic gestures
    dynamic_num_classes = len(torch.unique(dynamic_dataset.labels))

    dynamic_model = DynamicGestureModel(dynamic_num_classes)
    optimizer = torch.optim.Adam(dynamic_model.parameters(), lr=0.001)

    # Train and evaluate dynamic model
    trainer = GestureTrainer(dynamic_model, dynamic_data_loader, dynamic_data_loader, criterion, optimizer)
    trainer.train(num_epochs=50)
    trainer.evaluate()

    # Save dynamic model
    trainer.save_model('trained_data/dynamic_gesture_model.pth')