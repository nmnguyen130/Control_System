import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from src.modules.gesture_detection import *
from src.handlers.gesture_label_handler import GestureLabelHandler
from src.core.base_trainer import BaseTrainer

class GestureTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cpu'):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    def train(self, num_epochs=50, save_path=None, model_type=None):
        super().train(num_epochs, save_path, model_type)

    def _unpack_batch(self, batch):
        features, labels = batch
        return features, labels

def setup_static_model(label_handler):
    """Setup static gesture model, data loaders, and trainer."""
    static_csv_file = 'data/hand_gesture/static_data.csv'
    static_dataset = StaticGestureDataset(static_csv_file, label_handler=label_handler)
    
    train_data, val_data = train_test_split(
        static_dataset, 
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in static_dataset]
    )
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    static_num_classes = len(label_handler.get_all_static_gestures())
    static_model = StaticGestureModel(static_num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(static_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=5)

    return static_model, train_loader, val_loader, criterion, optimizer, scheduler

def setup_dynamic_model(label_handler):
    dynamic_csv_file = 'data/hand_gesture/dynamic_data.csv'
    dynamic_dataset = DynamicGestureDataset(dynamic_csv_file, label_handler=label_handler)
    
    train_data, val_data = train_test_split(
        dynamic_dataset, 
        test_size=0.2,
        random_state=42,
        stratify=[label for _, label in dynamic_dataset]
    )
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16, shuffle=False)

    dynamic_num_classes = len(label_handler.get_all_dynamic_gestures())
    dynamic_model = DynamicGestureModel(dynamic_num_classes)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(dynamic_model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    return dynamic_model, train_loader, val_loader, criterion, optimizer, scheduler

def main():
    label_handler = GestureLabelHandler('data/hand_gesture/static_labels.csv', 
                                 'data/hand_gesture/dynamic_labels.csv')
    
    # Setup and train static model
    # static_trainer = GestureTrainer(*setup_static_model(label_handler))
    # static_trainer.train(num_epochs=100, save_path='trained_data/static_gesture_model.pth', model_type='static')

    # Setup and train dynamic model
    dynamic_trainer = GestureTrainer(*setup_dynamic_model(label_handler))
    dynamic_trainer.train(num_epochs=70, save_path='trained_data/dynamic_gesture_model.pth', model_type='dynamic')

if __name__ == "__main__":
    main()