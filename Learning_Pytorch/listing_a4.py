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
    
torch.manual_seed(123)    
print()
model = NeuralNetwork(50, 3)
print(model)
print()
print(model.layers[0].weight.shape)
print()
print(model.layers[0].bias.shape)
print()
X = torch.rand((1,50))
with torch.no_grad():
    out = torch.softmax(model(X), dim=1)
    print(out)
print()

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print("Total number of trainable model parameters:", num_params)