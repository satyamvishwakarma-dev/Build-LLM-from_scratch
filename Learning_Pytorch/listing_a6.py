# defining a custom dataset class
import torch
from torch.utils.data import Dataset
import listing_a5 as dataset

class ToyDataset(Dataset):
    def __init__(self, X, Y):
        self.features = X
        self.labels = Y

    def __getitem__(self, index):
        one_X = self.features[index]
        one_Y = self.labels[index]
        return one_X, one_Y
    
    def __len__(self):
        return self.labels.shape[0]
    
train_ds = ToyDataset(dataset.X_train, dataset.y_train)
test_ds = ToyDataset(dataset.X_test, dataset.y_test)