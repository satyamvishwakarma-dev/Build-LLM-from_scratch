# Nural Network training in PyTorch

import torch
import torch.nn.functional as F

from Learning_Pytorch.listing_a4 import NeuralNetwork
from Learning_Pytorch.listing_a7 import train_loader

torch.manual_seed(123)
model = NeuralNetwork(num_input = 2, num_output=2)
optimizer = torch.optim.SGD(
    model.parameters(),
    lr = 0.5
)

num_epochs = 3
for epoch in range(num_epochs):
    model.train()

for batch_idx, (features, labels) in enumerate(train_loader):
    logits = model(features)

    loss = F.cross_entropy(logits, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # LOGGING
    print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
          f" | Batch {batch_idx+1:03d}/{len(train_loader):03d}"
          f" | Train Loss: {loss:.2f}")

if __name__ == "__main__":
    model.eval()