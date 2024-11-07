import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from data.data_processor import Dataprocessor
import copy
import matplotlib.pyplot as plt

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.best_loss = float('inf') 

    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0

        for data, target in train_loader:
            data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
            self.optimizer.zero_grad()

            # Add noise to input data
            data = Dataprocessor.add_noise(data)

            output = self.model(data)
            loss = self.criterion(output, target)

            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()

        return running_loss / len(train_loader)

    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(self.device), target.to(self.device, dtype=torch.int64)
                data = Dataprocessor.add_noise(data)
                output = self.model(data)
                loss = self.criterion(output, target)

                running_loss += loss.item()

        return running_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, num_epochs, best_model_save_path, final_model_save_path):
        """Train the model"""
        train_losses = []
        val_losses = []

        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            train_losses.append(train_loss)
            val_losses.append(val_loss)

            # Update learning rate based on validation loss
            self.scheduler.step(val_loss)

            print(f'Epoch {epoch + 1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')

            # Save best model
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                best_state = {
                    'State_dict': copy.deepcopy(self.model.state_dict()),
                    'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                    'scheduler': copy.deepcopy(self.scheduler.state_dict()),
                }
                torch.save(best_state, best_model_save_path)

            # Save final model after all epochs
            final_state = {
                'State_dict': copy.deepcopy(self.model.state_dict()),
                'optimizer': copy.deepcopy(self.optimizer.state_dict()),
                'scheduler': copy.deepcopy(self.scheduler.state_dict()),
            }
            torch.save(final_state, final_model_save_path)
            
            # Plot loss curves
            plt.figure()
            plt.plot(train_losses, label='Train Loss')
            plt.plot(val_losses, label='Validation Loss')
            plt.title('Loss Curves')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig('Loss.png', dpi=300)
            plt.close()