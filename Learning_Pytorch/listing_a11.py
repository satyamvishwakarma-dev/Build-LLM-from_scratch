import torch

from listing_a4 import NeuralNetwork
from listing_a7 import train_loader

torch.manual_seed(123)
model = NeuralNetwork(num_input = 2, num_output=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

num_epochs = 3

for epoch in range(num_epochs):

    model.train()
    for batch_idx, (features, labels) in enumerate(train_loader):   
        features, labels = features.to(device), labels.to(device)
        logits = model(features)
        loss = torch.nn.functional.cross_entropy(logits, labels)    # Loss Function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### LOGGING
        print(f"Epoch: {epoch+1:03d}/{num_epochs:03d}"
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train/Val Loss: {loss:.2f}")
        
    model.eval()