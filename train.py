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
from torch.optim.lr_scheduler import StepLR
from evaluate import evaluate

config = GPTConfig()
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

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
                "d_model": config.n_embed,
                "n_layers": config.n_layers,
                "context_length": config.block_size,
                "device": config.device,
                "dataset_source": "fine_web_edu_10T",
                "num_of_rows": config.take_samples
            }
        )
        logging.info("Weights and Biases tracking initialized.")
    except Exception as e:
        logging.warning(f"Failed to initialize Weights and Biases: {e}")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    fw= load_dataset(
        'arrow',
        data_files={"train":glob.glob('/scratch/prku/training_data/fineweb_edu_10B/train/*.arrow')[:5]},
        split="train",
        streaming=False,
    )
    # # Shuffle 
    shuffled_fw = fw.shuffle(seed=42)
    dataset_subset = shuffled_fw.take(config.take_samples)
    
    train_dataset = StreamingTextDataset(dataset_subset, tokenizer, config.block_size)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size)

    
    # --- Calculate Total Tokens ---
    logging.info("Calculating total tokens in the selected sample...")
    total_tokens = 0

    token_counts = dataset_subset.map(
        lambda x: {"len": [len(t) for t in tokenizer(x["text"], add_special_tokens=False)["input_ids"]]},
        batched=True,
        num_proc=os.cpu_count(), # Uses all available CPU cores
        remove_columns=dataset_subset.column_names # Drop text to save RAM
    )
    total_tokens = sum(token_counts["len"])    
    if wandb.run:
            wandb.config.update({"total_tokens": total_tokens})

    logging.info(f"Total tokens in the training sample: {total_tokens:,}")
    
    model = GPT(config)
    model=model.to(config.device)    
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    # scheduler = StepLR(optimizer, step_size=5000, gamma=0.5)

    # 7. Training Loop
    model.train()
    logging.info(f"Starting training for {config.epochs} epochs on ~{config.take_samples} documents per epoch...")
    last_checkpoint_path = None

    # Outer Epoch Loop
    for epoch in range(1, config.epochs + 1):
        step = 0        
        total_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}", mininterval=5)

        for batch in pbar:
            inputs = batch["input_ids"].to(config.device)
            targets = batch["target_ids"].to(config.device)

            logits, loss = model(inputs, targets)

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients to prevent spikes
            optimizer.step()
            # scheduler.step()

            current_loss = loss.item()
            total_loss += current_loss
            current_ppl = math.exp(current_loss)
            step += 1
            
            pbar.set_postfix(loss=f"{current_loss:.4f}", ppl=f"{current_ppl:.2f}")

            if wandb.run and step % 100 == 0:
                    wandb.log({
                        "train/loss": total_loss / step,
                        "train/perplexity": math.exp(total_loss / step),
                        "epoch": epoch
                    }, step=step)

            if step % 100 == 0:
                logging.info(f"Epoch {epoch} | Step {step} | Loss: {current_loss:.4f}|  PPL: {current_ppl:.2f}")
            
            # Optional: Save periodically
            
            if step % 5000 == 0:
                new_checkpoint_path = config.model_path+f"/checkpoint_epoch_{epoch}_step_{step}.pth"
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model': model,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': current_loss,
                }, new_checkpoint_path,_use_new_zipfile_serialization=False)
                logging.info(f"Checkpoint saved at step {step} to {new_checkpoint_path}")

                # 1. Delete previous model if it exists
                if last_checkpoint_path and os.path.exists(last_checkpoint_path):
                    try:
                        os.remove(last_checkpoint_path)
                        logging.info(f"Deleted previous checkpoint: {last_checkpoint_path}")
                    except Exception as e:
                        logging.warning(f"Could not delete old checkpoint: {e}")
                
                last_checkpoint_path = new_checkpoint_path

        # Calculate and log average loss for the completed epoch
        estimated_steps_per_epoch = config.take_samples // config.batch_size 
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
        'model': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': current_loss if 'current_loss' in locals() else None,
        'config': GPTConfig()
    }
    final_model_path=os.path.join(config.model_path,"completed_train.pth")
    torch.save(final_save_data, final_model_path,_use_new_zipfile_serialization=False)
    logging.info(f"Final model and optimizer saved to {config.model_path}")

    commonsense,arc_challenge=evaluate(final_model_path,config.device)
    wandb.log({
                    "CommonsenseQA accuracy": commonsense* 100,
                    "ARC Challenge accuracy": arc_challenge* 100})
    
    print(f"CommonsenseQA accuracy : {commonsense * 100:.2f}%")
    print(f"ARC Challenge accuracy : {arc_challenge * 100:.2f}%")

    if wandb.run:
        wandb.finish()

if __name__ == "__main__":
    train()