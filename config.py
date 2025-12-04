import torch

class GPTConfig:

    vocab_size = 50257 
    block_size = 256
    n_layers = 12
    n_heads = 12
    d_model = 768
    context_length = 1024
    dropout = 0.2
    
    # Training Hyperparameters
    batch_size = 16 #
    learning_rate = 3e-4
    epochs = 3
    
    # System
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
