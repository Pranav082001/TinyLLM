import torch
import torch.nn as nn
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.2):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        assert (n_heads * self.head_dim == d_model)

        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: torch.Tensor):
        B, seq_length, d_model = inputs.shape

        # Scaled Dot-Product Attention
        Q = self.query(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = self.key(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = self.value(inputs).view(B, seq_length, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled Dot-Product Attention
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # Applying mask to prevent attention to future tokens
        mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1).bool().to(inputs.device)
        attention_scores = attention_scores.masked_fill(mask, float('-inf'))
        
        attention_weights = torch.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(self.dropout(attention_weights), V)

        # Concatenating heads and put them back to the original shape
        attention_output = attention_output.permute(0, 2, 1, 3).contiguous()
        attention_output = attention_output.view(B, seq_length, d_model)

        out = self.fc_out(attention_output)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, context_length, d_model):
        super().__init__()

        #matrix of shape (context_length, d_model) to store the positional encodings
        pe = torch.zeros(context_length, d_model)

        #vector with positions [0, 1, 2, ..., context_length-1] of shape (context_length, 1)
        position = torch.arange(0, context_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0) # Shape: (1, context_length, d_model)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Slice the PE to the current sequence length of x
        return x + self.pe[:, :x.size(1), :]

import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.att = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model), # Named c_proj implicitly in init loop
            nn.Dropout(dropout)
        )
        # Rename the second linear for the init logic
        self.mlp[2].label = "c_proj" 

    def forward(self, x):
        # Pre-Norm: Residual stream stays "clean"
        x = x + self.att(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x    

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_length, dropout=0.2):
        super().__init__()
        self.context_length = context_length
        
        # 1. Embeddings
        self.wte = nn.Embedding(vocab_size, d_model)
        self.wpe = nn.Embedding(context_length, d_model) # Switched to Embedding for better learning
        self.dropout = nn.Dropout(dropout)
        
        # 2. Transformer Blocks (Pre-Norm)
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        
        # 3. Final LayerNorm (CRITICAL for Pre-Norm Architecture)
        self.ln_f = nn.LayerNorm(d_model)
        
        # 4. Output Head
        self.linear1 = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.wte.weight = self.linear1.weight

        # Initialize weights
        self.apply(self._init_weights)
        
        # Special scaling for residual projections
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                # Note: using n_layers here
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * n_layers))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, inputs, targets=None):
        device = inputs.device
        b, t = inputs.size()
        
        # Generate position indices
        pos = torch.arange(0, t, dtype=torch.long, device=device)
        
        # Token + Position Embeddings
        tok_emb = self.wte(inputs)
        pos_emb = self.wpe(pos)
        x = self.dropout(tok_emb + pos_emb)
        
        # Pass through blocks
        for block in self.blocks:
            x = block(x)
            
        # Apply final LayerNorm before the head
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.linear1(x)
            # SMART LOSS: Shift targets to predict the next token
            # Logits from [0...T-1] predict Targets [1...T]
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = targets[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        else:
            # Inference optimization: only calculate the last logit
            logits = self.linear1(x[:, [-1], :]) 
            loss = None
            
        return logits, loss

    @torch.no_grad()
    def generate(self, inputs, max_new_tokens):
        for _ in range(max_new_tokens):
            cond_inputs = inputs[:, -self.context_length:]
            logits, _ = self(cond_inputs)
            # forward already returns only the last logit if targets=None
            logits = logits[:, -1, :] 
            probs = torch.softmax(logits, dim=-1)            
            idx_next = torch.multinomial(probs, num_samples=1) 
            inputs = torch.cat([inputs, idx_next], dim=1)
        return inputs