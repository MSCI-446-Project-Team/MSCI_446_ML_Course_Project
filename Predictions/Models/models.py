import torch
import torch.nn as nn


class TimeSeriesNeuralNetwork(nn.Module):
    def __init__(self, sequence_length, num_features, output_size):
        super().__init__()
        self.flatten = nn.Flatten()
        input_size = sequence_length * num_features
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, output_size),
        )

    def forward(self, x):
        x = self.flatten(x)
        output = self.linear_relu_stack(x)
        return output


class LSTMTimeSeriesModel(nn.Module):
    def __init__(self, sequence_length, num_features, output_size, hidden_dim, num_layers, dropout_rate):
        super().__init__()
        self.num_features = num_features
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=num_features,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout_rate)
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):

        h0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)

        c0 = torch.zeros(self.num_layers, x.size(0),
                         self.hidden_dim).to(x.device)

        lstm_out, _ = self.lstm(x, (h0, c0))

        y_pred = self.linear(lstm_out[:, -1, :])
        return y_pred
