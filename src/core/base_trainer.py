import os
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt

class BaseTrainer:
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler=None, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_val_accuracy = 0

        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def train(self, num_epochs, save_path=None, model_type=None):
        for epoch in range(num_epochs):
            train_loss, train_acc = self._train_one_epoch(epoch, num_epochs)
            val_loss, val_acc = self.validate()

            # Store the loss and accuracy for plotting
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)

            if self.scheduler:
                self.scheduler.step(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}] - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            if val_acc > self.best_val_accuracy:
                self.best_val_accuracy = val_acc
                if save_path:
                    self.save_model(save_path, model_type)
                print(f"New best {model_type} model saved with validation accuracy: {val_acc:.2f}%")

            # After training, plot the results
            self.plot_training_history(model_type)

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

    def save_model(self, file_path, model_type):
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            torch.save(self.model.state_dict(), file_path)
            print(f"{model_type} model saved to {file_path}")
        except Exception as e:
            print(f"Failed to save {model_type} model: {e}")

    def plot_training_history(self, model_type):
        epochs = range(1, len(self.train_losses) + 1)

        plt.figure(figsize=(12, 5))

        # Plot training and validation loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_losses, label=f'Training Loss ({model_type})')
        plt.plot(epochs, self.val_losses, label=f'Validation Loss ({model_type})')
        plt.title(f'Training and Validation Loss ({model_type})')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        # Plot training and validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_accuracies, label=f'Training Accuracy ({model_type})')
        plt.plot(epochs, self.val_accuracies, label=f'Validation Accuracy ({model_type})')
        plt.title(f'Training and Validation Accuracy ({model_type})')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy (%)')
        plt.legend()

        plt.tight_layout()

        # Save the plots to the trained_data directory
        save_dir = 'trained_data'
        os.makedirs(save_dir, exist_ok=True)  # Create the directory if it doesn't exist
        plt.savefig(os.path.join(save_dir, f'training_history_{model_type}.png'))

        plt.close()  # Close the figure to free memory
