import torch
from torch.utils.data import Dataset

import torch
from torch.utils.data import IterableDataset

class StreamingTextDataset(IterableDataset):
    def __init__(self, hf_dataset, tokenizer, block_size):
        self.hf_dataset = hf_dataset
        self.tokenizer = tokenizer
        self.block_size = block_size
        self.eos_id = tokenizer.eos_token_id

    def __iter__(self):
        """
        Iterates through the streaming dataset, tokenizes text, 
        and yields chunks of size (block_size + 1).
        """
        buffer = []
        
        for sample in self.hf_dataset:
            # 1. Get text and tokenize
            text = sample['text']
            tokens = self.tokenizer.encode(text)
            tokens.append(self.eos_id) # Add EOS token at the end of docs
            
            buffer.extend(tokens)
            
            # 2. Yield chunks while buffer has enough data
            while len(buffer) >= self.block_size + 1:
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size + 1:] # Slide window (non-overlapping)
                
                # Create input (x) and target (y)
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                
                yield {
                    "input_ids": x,
                    "target_ids": y
                }