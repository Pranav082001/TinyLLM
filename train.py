import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import os
from datasets import load_dataset

from config import GPTConfig
from model import GPT
from dataset import TextDataset

def train():
    config = GPTConfig()
    print(f"Using device: {config.device}")

    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    
    # In production, load a file: with open('input.txt', 'r') as f: text = f.read()
    fw = load_dataset(
        "HuggingFaceFW/fineweb",
        name="CC-MAIN-2025-26",
        split="train",
        streaming=True,
        columns=["text"],
    )
    
    dataset = TextDataset(fw, tokenizer, block_size=config.block_size)
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} samples")

    # 4. Initialize Model
    model = GPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        context_length=config.context_length
    ).to(config.device)
    
    # Compile for speed (Python 3.10+ and Linux recommended)
    try:
        model = torch.compile(model)
        print("Model compiled.")
    except Exception as e:
        print(f"Could not compile model: {e}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    # 5. Training Loop
    model.train()
    for e in range(config.epochs):
        total_loss = 0
        for i, batch in enumerate(train_loader):
            inputs = batch["input_ids"].to(config.device)
            targets = batch["target_ids"].to(config.device)

            # Forward pass (passing targets ensures loss is calculated)
            logits, loss = model(inputs, targets)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if i % 10 == 0:
                print(f"Epoch {e} | Batch {i} | Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"--> Epoch {e} finished. Avg Loss: {avg_loss:.4f}")
        
        # Save checkpoint
        torch.save(model.state_dict(), config.model_path)
        print(f"Model saved to {config.model_path}")

    # 6. Simple Generation Test
    print("\nGenerating text...")
    model.eval()
    start_context = "Hello world"
    input_ids = tokenizer.encode(start_context, return_tensors='pt').to(config.device)
    
    generated_ids = model.generate(input_ids, max_new_tokens=50)
    decoded_text = tokenizer.decode(generated_ids[0].tolist())
    print(f"Result:\n{decoded_text}")

if __name__ == "__main__":
    train()