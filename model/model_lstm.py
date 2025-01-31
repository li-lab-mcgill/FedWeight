import torch
import torch.nn as nn


class LSTM(nn.Module):
    def __init__(self, input_dim):
        super(LSTM, self).__init__()
        self.num_layers = 1
        self.hidden_dim = 16
        self.lstm = nn.LSTM(input_dim, self.hidden_dim, self.num_layers, batch_first=True)
        self.fc = nn.Linear(self.hidden_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # N x time_window x hidden_dim
        out = self.fc(lstm_out)[:, :, 0]  # N x time_window
        return torch.sigmoid(out)
