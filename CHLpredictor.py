import torch
import torch.nn as nn

class CHLpredictor(nn.Module):

    def __init__(self, device, input_size, hidden_size, num_stacked_layers, out_features=1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_stacked_layers = num_stacked_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_stacked_layers,
                            batch_first=True)

        self.fc = nn.Linear(hidden_size, out_features)
        self.learning_rate = 0.001
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.lstm.parameters(), lr=self.learning_rate)
        self.device = device
        self.lstm.to(self.device)

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)
        c0 = torch.zeros(self.num_stacked_layers, batch_size, self.hidden_size).to(self.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

