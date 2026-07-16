# Nural Network training in PyTorch

import torch
import torch.nn.functional as F

from listing_a4 import NeuralNetwork
from listing_a5 import X_train
from listing_a5 import y_train
from listing_a7 import train_loader

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
              f" | Batch {batch_idx:03d}/{len(train_loader):03d}"
              f" | Train Loss: {loss:.2f}")
        
def compute_accuracy(model, data_loader):
    model = model.eval()
    correct = 0.0
    total_examples = 0

    for idx, (features, labels) in enumerate(data_loader):
        with torch.no_grad():
            logits = model(features)

        pridiction = torch.argmax(logits, dim=1)
        compare = labels == pridiction
        correct += torch.sum(compare)
        total_examples += len(labels)

    return (correct / total_examples).item() # type: ignore


if __name__ == "__main__":
    model.eval()
    print("---")
    with torch.no_grad():
        output = model(X_train)
    print(output)
    print("---")

    torch.set_printoptions(sci_mode=False)
    probas = torch.softmax(output, dim=1)
    print(probas)

    print("---")

    pridiction = torch.argmax(probas, dim=1)
    print(pridiction)
   
    print("---")
    pridiction = torch.argmax(output, dim=1)
    print(pridiction)

    print("---")
    print(pridiction == y_train)
    print("---")
    print(compute_accuracy(model, train_loader))

    # print(model)
    # This gives the total number of parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total Parameters: {total_params}")