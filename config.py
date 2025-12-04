import torch

class GPTConfig:

    vocab_size = 50257 
    block_size = 1024
    n_layers = 12
    n_heads = 12
    d_model = 768
    context_length = 1024
    dropout = 0.2
    
    # Training Hyperparameters
    batch_size = 16 #
    learning_rate = 3e-4
    epochs = 3

    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_subset = "CC-MAIN-2025-26"
    take_samples=2
    model_path="/nethome/prku/pretraining_llm_group1/TinyLLM/model"
    # System
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
