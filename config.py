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
    batch_size = 32 #
    learning_rate = 3e-4
    epochs = 3

    dataset_name = "HuggingFaceFW/fineweb-edu"
    dataset_subset = "CC-MAIN-2025-26"
    logfile_name="baseline_training.log"
    take_samples=1000
    model_path="/nethome/prku/pretraining_llm_group1/TinyLLM/models/baseline_trained.pth"
    # System
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


#LOcal Config

# class GPTConfig:

#     vocab_size = 50257 
#     block_size = 256
#     n_layers = 12
#     n_heads = 12
#     d_model = 768
#     context_length = 1024
#     dropout = 0.2
    
#     # Training Hyperparameters
#     batch_size = 4 #
#     learning_rate = 3e-4
#     epochs = 3

#     dataset_name = "HuggingFaceFW/fineweb-edu"
#     dataset_subset = "CC-MAIN-2025-26"
#     logfile_name="baseline_training.log"
#     take_samples=10
#     model_path="/Users/pranavk/Desktop/Saarland/Sem 3/Pretraining LM/models/baseline_trained.pth"
#     device = torch.device("mps")
