import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class IntentModel(nn.Module):
    def __init__(self, input_size, hidden_layers=None, output_size=None):
        super(IntentModel, self).__init__()

        layers = []
        in_features = input_size
        
        # Add hidden layers if specified
        if hidden_layers is None:
            layers.append(nn.Linear(in_features, 128))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
            layers.append(nn.Linear(128, 64))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        else:
            for hidden in hidden_layers:
                layers.append(hidden)
        
        layers.append(nn.Linear(in_features=64, out_features=output_size))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)