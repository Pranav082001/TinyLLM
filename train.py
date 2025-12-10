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
import math,time,glob
import wandb
from api_keys import wandb_api

config = GPTConfig()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler(config.logfile_name),
        logging.StreamHandler()
    ]
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wandb.login(key=wandb_api)
def train():
    config = GPTConfig()
    logging.info(f"Using device: {config.device}")
    try:
        wandb.init(
            project="Pretraining_LLM",
            name=f"run-{time.strftime('%Y%m%d-%H%M%S')}",
            config={
                "learning_rate": config.learning_rate,
                "batch_size": config.batch_size,
                "epochs": config.epochs,
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "context_length": config.context_length,
                "device": config.device,
                "dataset_source": "fine_web_edu_10T",
                "sample_size_rows": config.take_samples
            }
        )
        logging.info("Weights and Biases tracking initialized.")
    except Exception as e:
        logging.warning(f"Failed to initialize Weights and Biases: {e}")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

    logging.info(f"Loading {config.dataset_name} ({config.dataset_subset}) in streaming mode...")
    # fw = load_dataset(
    #     config.dataset_name,
    #     name=config.dataset_subset,
    #     split="train",
    #     streaming=True,
    #     columns=["text"],
    # )
    fw= load_dataset(
        'arrow',
        data_files={"train":glob.glob('/nethome/prku/pretraining_llm_group1/training_data/fineweb_edu_10B/train/*.arrow')[:3]},
        split="train",
        streaming=False,
    )
    # # Shuffle and Take 10%
    shuffled_fw = fw.shuffle(seed=42)
    dataset_subset = shuffled_fw.take(config.take_samples)
    
    train_dataset = StreamingTextDataset(fw, tokenizer, config.block_size)
    
    # DataLoader for IterableDatasets usually requires num_workers=0 or special handling
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
    
    # --- Calculate Total Tokens ---
    logging.info("Calculating total tokens in the selected sample...")
    total_tokens = 0
    if wandb.run:
        wandb.config.update({"total_tokens": total_tokens})

    for example in tqdm(dataset_subset, desc="Tokenizing Sample"):
        # The tokenizer's encode method gives the token IDs (list of integers)
        tokens = tokenizer.encode(example['text'], add_special_tokens=False)
        total_tokens += len(tokens)
    
    logging.info(f"Total tokens in the training sample: {total_tokens:,}")
    
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
    logging.info(f"Starting training for {config.epochs} epochs on ~{config.take_samples} documents per epoch...")
    
    step = 0
    
    # Outer Epoch Loop
    for epoch in range(1, config.epochs + 1):
        logging.info("=" * 60)
        logging.info(f"--- Starting Epoch {epoch}/{config.epochs} ---")
        logging.info("=" * 60)
        
        total_loss = 0.0
        
        # NOTE: Re-create dataset and loader for each epoch to restart the stream
        train_dataset = StreamingTextDataset(dataset_subset, tokenizer, config.block_size)
        
        # DataLoader for IterableDatasets usually requires num_workers=0 or special handling
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size)
        
        # Calculate estimated steps per epoch for the progress bar
        estimated_steps_per_epoch = config.take_samples // config.batch_size 
        pbar = tqdm(
            train_loader, 
            total=estimated_steps_per_epoch,
            desc=f"Epoch {epoch}/{config.epochs}", 
            mininterval=5
        )

        for batch in pbar:
            inputs = batch["input_ids"].to(config.device)
            targets = batch["target_ids"].to(config.device)

            logits, loss = model(inputs, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            current_ppl = math.exp(current_loss)
            step += 1
            
            pbar.set_postfix(loss=f"{current_loss:.4f}", ppl=f"{current_ppl:.2f}")

            if wandb.run and step % 10 == 0:
                wandb.log({
                    "train/loss": total_loss / step,
                    "train/perplexity": math.exp(total_loss / step),
                    "epoch": epoch
                }, step=step)

            if step % 20 == 0:
                logging.info(f"Epoch {epoch} | Step {step} | Loss: {current_loss:.4f}|  PPL: {current_ppl:.2f}")
            
            # Optional: Save periodically
            if step % 2000 == 0:
                checkpoint_path = f"/models/checkpoint_epoch_{epoch}_step_{step}.pth"
                # --- Save Model and Optimizer State ---
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                }, checkpoint_path,_use_new_zipfile_serialization=False)
                logging.info(f"Checkpoint saved at step {step} to {checkpoint_path}")

        # Calculate and log average loss for the completed epoch
        steps_in_epoch = step - (config.epochs - 1) * estimated_steps_per_epoch
        avg_epoch_loss = total_loss / steps_in_epoch if steps_in_epoch > 0 else 0
        avg_epoch_ppl = math.exp(avg_epoch_loss)
        if wandb.run:
            wandb.log({
                "epoch_avg/loss": avg_epoch_loss,
                "epoch_avg/perplexity": avg_epoch_ppl,
                "epoch": epoch})

        logging.info(f"--- Epoch {epoch} complete. Average Loss: {avg_epoch_loss:.4f}  | Average PPL: {avg_epoch_ppl:.2f} ---")


    print(f"Training finished after {config.epochs} epochs.")
    # --- Final Save Model and Optimizer State ---
    final_save_data = {
        'epoch': config.epochs,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss if 'current_loss' in locals() else None,
    }
    torch.save(final_save_data, config.model_path,_use_new_zipfile_serialization=False)
    logging.info(f"Final model and optimizer saved to {config.model_path}")
    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    train()