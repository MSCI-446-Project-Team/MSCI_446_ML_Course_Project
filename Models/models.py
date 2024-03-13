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
