import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from datasets import load_dataset
import os

from config import GPTConfig
from model import GPT
from dataset import StreamingTextDataset
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler()
    ]
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train():
    config = GPTConfig()
    logging.info(f"Using device: {config.device}")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    logging.info(f"Loading {config.dataset_name} ({config.dataset_subset}) in streaming mode...")
    fw = load_dataset(
        config.dataset_name,
        name=config.dataset_subset,
        split="train",
        streaming=True,
        columns=["text"],
    )
    
    # # Shuffle and Take 10%
    shuffled_fw = fw.shuffle(seed=42)
    dataset_subset = shuffled_fw.take(config.take_samples)
    
    train_dataset = StreamingTextDataset(fw, tokenizer, config.block_size)
    
    # DataLoader for IterableDatasets usually requires num_workers=0 or special handling
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    
    # 6. Initialize Model
    model = GPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        context_length=config.context_length
    ).to(config.device)
    
    # try:
    #     model = torch.compile(model)
    #     logging.info("Model compiled successfully.")
    # except Exception as e:
    #     logging.warning(f"Could not compile model: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 7. Training Loop
    model.train()
    logging.info(f"Starting training on ~{config.take_samples} documents...")
    
    step = 0
    total_loss = 0
    
    for batch in tqdm(train_loader):
        inputs = batch["input_ids"].to(config.device)
        targets = batch["target_ids"].to(config.device)

        logits, loss = model(inputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        step += 1

        if step % 10 == 0:
            logging.info(f"Step {step} | Loss: {loss.item():.4f}")
        
        # Optional: Save periodically
        if step % 1000 == 0:
            torch.save(model.state_dict(), config.model_path)
            logging.info(f"Checkpoint saved at step {step}")

    print(f"Training finished. Final Loss: {loss.item():.4f}")
    torch.save(model.state_dict(), config.model_path)

if __name__ == "__main__":
    train()