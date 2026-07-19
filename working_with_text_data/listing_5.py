import torch
from torch.utils.data import Dataset, dataloader

class GPTDataset(Dataset):
    def __init__(self, txt, tokenizer, max_length):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)