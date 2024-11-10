import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.modules.voice_assistant.intent_dataset import IntentDataset
from src.modules.voice_assistant.intent_model import IntentModel
from tqdm import tqdm
from nltk.stem import WordNetLemmatizer

class SpeechTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_accuracy = 0

    def train(self, num_epochs, save_path=None):
        for epoch in range(num_epochs):
            train_loss, train_acc = self._train_one_epoch(epoch, num_epochs)
            val_loss, val_acc = self.validate()

            if self.scheduler:
                self.scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                if save_path:
                    self.save_model(save_path)
                print(f"New best model saved with validation accuracy: {val_acc:.2f}%")

    def _train_one_epoch(self, epoch, num_epochs):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        progress = tqdm(self.train_loader, desc=f"Epoch [{epoch + 1}/{num_epochs}]")

        for data, target in progress:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(self.train_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                loss = self.criterion(output, target)
                total_loss += loss.item()

                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = total_loss / len(self.val_loader)
        accuracy = 100.0 * correct / total
        return avg_loss, accuracy

    def save_model(self, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            torch.save(self.model.state_dict(), file_path)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
    
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
    optimizer = Adam(model.parameters(), lr=0.01)
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
    trainer.train(num_epochs=100, save_path='trained_data/intent_model.pth')

if __name__ == '__main__':
    main()