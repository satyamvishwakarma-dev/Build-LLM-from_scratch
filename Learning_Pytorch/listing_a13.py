import torch
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import torch.multiprocessing as mp
import os
from torch.utils.data import DataLoader
from listing_a9 import compute_accuracy_ddp
from listing_a6 import train_ds
from listing_a7 import test_loader
from listing_a4 import NeuralNetwork

def ddp_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size

    )
    torch.cuda.set_device(rank)

def prepare_dataset():
    # insert dataset preparation code
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,
        pin_memory=True,
        drop_last=True,
        sampler=DistributedSampler(train_ds)
    )
    return train_loader, test_loader

def main(rank, world_size, num_epochs):
    ddp_setup(rank, world_size)
    train_loader, test_loader = prepare_dataset()
    model = NeuralNetwork(num_input=2, num_output=2)
    model.to(rank)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
    model = DDP(model, device_ids=[rank])
    for epoch in range(num_epochs):
        for features, labels in train_loader:
            features, labels = features.to(rank), labels.to(rank)
            # 1. Forward pass
            outputs = model(features)
            
            # 2. Compute loss (make sure you use your actual loss function name here)
            # If your book uses CrossEntropyLoss, it might look like: loss_fn(outputs, labels)
            loss = torch.nn.functional.cross_entropy(outputs, labels) 
            
            # 3. Backward pass & optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # insert model prediction and backpropagation code
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")
                  
    model.eval()
    train_acc = compute_accuracy_ddp(model, train_loader, rank)
    print(f"[GPU{rank}] Traning Accuracy", train_acc)
    test_acc = compute_accuracy_ddp(model, test_loader, rank)
    print(f"[GPU{rank}] Testing Accuracy", test_acc)
    destroy_process_group()

if __name__ == "__main__":
    print("Number of GPUs Available: ", torch.cuda.device_count())
    torch.manual_seed(123)
    num_epochs = 3
    world_size = torch.cuda.device_count()
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size) # type: ignore


