import torch
import os

class GPTConfig:

    block_size=512  # max sequence length 
    vocab_size=50257
    n_layers=12
    n_head=12
    n_embed=768
    dropout=0.2
    
    # Training Hyperparameters
    batch_size = 8 #
    learning_rate = 5e-5
    epochs = 1
    
    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_subset = "CC-MAIN-2025-26"
    take_samples=100000
    model_path="/scratch/prku/models/hyperparameter_tuning/LR_warmup_with_cosin_annealing"
    if os.path.isdir(model_path):
        pass
    else :
        os.mkdir(model_path)
    # System
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
   
