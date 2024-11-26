import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.modules.voice_assistant.intent_dataset import IntentDataset
from src.modules.voice_assistant.intent_model import IntentModel
from src.core.base_trainer import BaseTrainer

class SpeechTrainer(BaseTrainer):
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cpu'):
        super().__init__(model, train_loader, val_loader, criterion, optimizer, scheduler, device)

    def train(self, num_epochs, save_path=None, model_type=None):
        super().train(num_epochs, save_path, model_type)

def setup_speech_model():
    # Create data loaders
    intent_json_path = 'data/speech/intents.json'
    train_dataset = IntentDataset(intent_json_path)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    input_size = len(train_dataset.words)
    output_size = len(train_dataset.intents)

    # Initialize model, criterion, optimizer, and scheduler
    model = IntentModel(input_size, output_size=output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=1e-3)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
        
    return model, train_loader, criterion, optimizer, scheduler

def main():
    model, train_loader, criterion, optimizer, scheduler = setup_speech_model()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    trainer = SpeechTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )
    trainer.train(num_epochs=100, save_path='trained_data/intent_model.pth', model_type='intents')

if __name__ == '__main__':
    main()