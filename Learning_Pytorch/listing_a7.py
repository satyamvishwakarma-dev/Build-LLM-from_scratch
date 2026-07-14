# Instaiating data loaders
import torch
from torch.utils.data import DataLoader
import listing_a6 as toy_dataset

torch.manual_seed(123)

train_loader = DataLoader(
    dataset = toy_dataset.train_ds,
    batch_size=2,
    shuffle = True,
    num_workers=0
)

test_loader = DataLoader(
    dataset = toy_dataset.test_ds,
    batch_size=2,
    shuffle = False,
    num_workers=0
)

for idx, (x, y) in enumerate(train_loader):
    print(f"Batch {idx+1}:", x, y)