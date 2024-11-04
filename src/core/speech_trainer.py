import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.modules.voice_assistant.speech_dataset import SpeechDataset
from src.modules.voice_assistant.speech_model import SpeechRecognitionModel
from tqdm import tqdm

# ERROR: 
class SpeechTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        
    def train(self, num_epochs):
        best_val_loss = float('inf')
        
        for epoch in range(num_epochs):
            # Training phase
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(tqdm(self.train_loader)):
                data, target = data.to(self.device), target.to(self.device)
                
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
                
            # Validation phase
            val_loss, val_acc = self.evaluate()
            
            # Print epoch results
            print(f'Epoch: {epoch+1}/{num_epochs}')
            print(f'Training Loss: {train_loss/len(self.train_loader):.4f}')
            print(f'Training Accuracy: {100.*train_correct/train_total:.2f}%')
            print(f'Validation Loss: {val_loss:.4f}')
            print(f'Validation Accuracy: {val_acc:.2f}%')
            
            # Learning rate scheduling
            if self.scheduler:
                self.scheduler.step(val_loss)
                
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'trained_data/best_speech_model.pth')
                
    def evaluate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        return val_loss/len(self.val_loader), 100.*correct/total
    
def setup_speech_model():
    # Create data loaders
    train_dataset = SpeechDataset(csv_file='data/speech/commands.csv', audio_dir='data/speech')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    
    # Determine the number of classes
    num_classes = len(train_dataset.command_to_label)
    
    # Initialize model, criterion, optimizer, and scheduler
    model = SpeechRecognitionModel(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
        
    return model, train_loader, criterion, optimizer, scheduler

def main():
    trainer = SpeechTrainer(*setup_speech_model())
    trainer.train(num_epochs=10)

if __name__ == '__main__':
    main()