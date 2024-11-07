import torch
import torch.nn as nn

class LSTMWakewordModel(nn.Module):
    def __init__(self, num_classes, feature_size, hidden_size, num_layers, dropout, bidirectional, device='cpu'):
        super(LSTMWakewordModel, self).__init__()
        self.num_classes = num_classes
        self.feature_size = feature_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(input_size=self.feature_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, batch_first=True, dropout=self.dropout,
                            bidirectional=self.bidirectional)
        self.classifier = nn.Linear(self.hidden_size * self.num_directions, self.num_classes)

    def __init_hidden(self, batch_size):
        return (torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device),
                torch.zeros(self.num_layers * self.num_directions, batch_size, self.hidden_size).to(self.device))

    def forward(self, x):
        x = self.layernorm(x)
        hidden = self.__init_hidden(x.size()[-1])
        out, (hn, cn) = self.lstm(x, hidden)
        out = self.classifier(hn)
        return out
