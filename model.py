from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn import functional as F
import torch 
import torch.nn as nn
import math
from torch.nn import functional as F

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embed),
            wpe = nn.Embedding(config.block_size, config.n_embed),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.n_embed),
        ))

        self.lm_head = nn.Linear(config.n_embed, config.vocab_size, bias=False)

    def forward(self, idx, targets=None):
        # idx is of shape (B, T)
        B, T = idx.size()

        pos = torch.arange(0, T, dtype=torch.long, device=idx.device) # shape (T)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (T, n_embd)
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (B, T, n_embd)
        x = tok_emb + pos_emb

        # forward pass in block 
        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
        loss = None

        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            
        return logits, loss
    

class Block(nn.Module):
    def __init__(self,config):

        super().__init__()
        self.ln_1=nn.LayerNorm(config.n_embed)
        self.attn=MultiHeadAttention(config)
        self.ln_2=nn.LayerNorm(config.n_embed)
        self.mlp=MLP(config)

    def forward(self,x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x
    

class MLP( nn.Module):
    def __init__(self,config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embed, 4*config.n_embed)
        self.gelu = nn.GELU(approximate='tanh') #approximate gelu is faster, also used in original gpt2
        self.c_proj = nn.Linear( 4*config.n_embed, config.n_embed)
        
    def forward  (self,x):
        x=self.c_fc(x)
        x=self.gelu(x)
        x=self.c_proj(x)

        return x


class Head(nn.Module):
    def __init__(self, config):
        super().__init__()
        head_size = config.n_embed // config.n_head
        # GPT-2 uses biases in these layers
        self.query = nn.Linear(config.n_embed, head_size, bias=True)
        self.key   = nn.Linear(config.n_embed, head_size, bias=True)
        self.value = nn.Linear(config.n_embed, head_size, bias=True)
        
        self.register_buffer('tril', torch.tril(torch.ones(config.block_size, config.block_size)))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, head_size)
        q = self.query(x) # (B, T, head_size)
        
        # Compute attention scores
        # (B, T, head_size) @ (B, head_size, T) -> (B, T, T)
        wei = q @ k.transpose(-2, -1) * (k.shape[-1]**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        
        v = self.value(x) # (B, T, head_size)
        out = wei @ v     # (B, T, head_size)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = nn.ModuleList([Head(config) for _ in range(config.n_head)])
        # The projection layer back to n_embed
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        # Concatenate all head outputs
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.c_proj(out))
        return out
        