import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from src.modules.voice_assistant.wakework_dataset import WakeWordData
from src.modules.voice_assistant.wakework_model import LSTMWakewordModel
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

    def train_one_epoch(self):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for data, target in tqdm(self.train_loader, desc="Training"):
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        avg_loss = train_loss / len(self.train_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def validate(self):
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc="Validating"):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                val_loss += self.criterion(output, target).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

        avg_loss = val_loss / len(self.val_loader)
        accuracy = 100. * correct / total
        return avg_loss, accuracy

    def train(self, num_epochs, save_path=None):
        best_val_loss = float('inf')

        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')
            train_loss, train_accuracy = self.train_one_epoch()
            val_loss, val_accuracy = self.validate()

            print(f'Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy:.2f}%')
            print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

            if self.scheduler:
                self.scheduler.step(val_loss)

            # Save the best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model(save_path)
                print(f'Saved best model to {save_path}')

    def save_model(self, file_path):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            torch.save(self.model.state_dict(), file_path)
            print(f"Model saved to {file_path}")
        except Exception as e:
            print(f"Failed to save model: {e}")
    
def setup_speech_model():
    # Create data loaders
    train_dataset = WakeWordData(csv_file='data/speech/commands.csv', audio_dir='data/speech', target_length=128)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Determine the number of classes
    num_classes = len(train_dataset.command_to_label)
    
    # Initialize model, criterion, optimizer, and scheduler
    model_params = {
        "num_classes": 1, "feature_size": 40, "hidden_size": 128,
        "num_layers": 1, "dropout" :0.1, "bidirectional": False
    }
    model = LSTMWakewordModel(**model_params, device='cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)
        
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
    trainer.train(num_epochs=10, save_path='trained_data/wake_word_model.pth')

if __name__ == '__main__':
    main()