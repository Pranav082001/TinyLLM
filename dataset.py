import torch
from torch.utils.data import Dataset

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, block_size):
        self.tokenizer = tokenizer
        # Encode the entire text at once
        self.data = tokenizer.encode(text, return_tensors='pt').squeeze()
        self.block_size = block_size

    def __len__(self):
        # We can extract len(data) - block_size samples
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Chunk the data
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # Inputs: 0 to N-1
        x = chunk[:-1]
        # Targets: 1 to N (shifted by 1)
        y = chunk[1:]
        
        return {
            "input_ids": x,
            "target_ids": y
        }