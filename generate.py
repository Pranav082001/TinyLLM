import torch
from transformers import AutoTokenizer
import argparse
import time
import logging
import os

from config import GPTConfig
from model import GPT

def generate_text(model, tokenizer, prompt, max_new_tokens=50, temperature=1.0, top_k=50):
    """
    Generates text by autocompleting a given prompt.
    """
    config = GPTConfig()
    device=config.device

    model.eval()
    encoded_prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
    generated_ids = encoded_prompt
        
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop the context to the model's max context length
            input_context = generated_ids[:, -model.context_length:]

            # Forward pass to get logits for the next token
            logits, _ = model(input_context)
            
            # Focus on the last token's logits (the prediction for the next token)
            next_token_logits = logits[:, -1, :] 
            
            # Apply temperature (softening or sharpening probabilities)
            if temperature == 0.0:
                # Deterministic argmax if temperature is 0
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                # Apply temperature and Top-K sampling
                next_token_logits = next_token_logits / temperature
                
                # Apply Top-K filtering
                v, _ = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                # Set all logits below the k-th element to a very low value (negative infinity)
                next_token_logits[next_token_logits < v[:, [-1]]] = -float('Inf')
                
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            
            # GPT-2 typically uses the <|endoftext|> token (id=50256 or similar) but let's check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Append the new token to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token], dim=1)

    # 3. Decode and return
    generated_text = tokenizer.decode(generated_ids.squeeze(0).tolist(), skip_special_tokens=True)
    return generated_text

def main():
    parser = argparse.ArgumentParser(description="Test a custom-trained GPT model for text generation.")
    parser.add_argument("--prompt", type=str, required=True, help="The initial text prompt to complete.")
    parser.add_argument("--max_tokens", type=int, default=100, help="Maximum number of tokens to generate.")
    parser.add_argument("--temp", type=float, default=0.9, help="Sampling temperature (0.0 for greedy).")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K sampling limit.")
    
    args = parser.parse_args()

    # --- Configuration and Setup ---
    config = GPTConfig()
    
    # Check for CUDA device
    device = config.device
    
    logging.info(f"Using device for generation: {config.device}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    
    # Initialize the model structure
    model = GPT(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        context_length=config.context_length
    ).to(config.device)

    # Load the trained weights
    if not os.path.exists(config.model_path):
        logging.error(f"Model weights not found at {config.model_path}. Please check your config.py and training output.")
        return
        
    print(f"Loading weights from {config.model_path}...")
    checkpoint = torch.load(config.model_path, map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Model weights loaded successfully.")

    # --- Run Generation ---
    start_time = time.time()
    
    generated_text = generate_text(
        model, 
        tokenizer, 
        args.prompt, 
        max_new_tokens=args.max_tokens, 
        temperature=args.temp, 
        top_k=args.top_k
    )
    
    end_time = time.time()
    
    # --- Output Results ---
    print("\n" + "="*50)
    print("      GENERATION COMPLETE")
    print("="*50)
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    print("-" * 50)
    print(f"Prompt: {args.prompt}")
    print("-" * 50)
    print("Generated Text:")
    print(generated_text)
    print("="*50 + "\n")


if __name__ == "__main__":
    main()