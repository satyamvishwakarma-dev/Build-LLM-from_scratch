from typing import Any

import torch


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_input, num_output):
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_input, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            
            # output layer
            torch.nn.Linear(20, num_output)
        )

    def forward(self, x):
        logits = self.layers(x)
        return logits
    
model = NeuralNetwork(50, 3)
print(model)