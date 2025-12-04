import torch
import torch.nn as nn
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

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.2):
        super().__init__()
        self.att = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.fcn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, logits):
        att_logits = self.att(logits)
        adn_logits = self.ln1(logits + att_logits)
        logits = self.dropout(adn_logits)
        logits = self.fcn(logits)
        logits = self.ln2(logits + adn_logits)
        return logits

class GPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, context_length):
        super().__init__()
        self.context_length = context_length
        self.wte = nn.Embedding(vocab_size, d_model) # word token embeddings
        self.wpe = PositionalEncoding(context_length, d_model) # word position encodings
        self.blocks = nn.ModuleList([GPTBlock(d_model, n_heads) for _ in range(n_layers)])
        self.linear1 = nn.Linear(d_model, vocab_size)

        # Weight tying (optional but standard in GPT)
        self.wte.weight = self.linear1.weight

    def forward(self, inputs, targets=None):
        logits = self.wte(inputs)  # dim -> batch_size, sequence_length, d_model
        logits = self.wpe(logits)
        
        for block in self.blocks:
            logits = block(logits)
            
        logits = self.linear1(logits)
        
        loss = None
        if targets is not None:
            batch_size, sequence_length, d_model = logits.shape
            logits_reshaped = logits.view(batch_size * sequence_length, -1) # d_model is vocab_size here
            targets_reshaped = targets.view(batch_size * sequence_length)
            loss = torch.nn.functional.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss
    
    @torch.no_grad()
    def generate(self, inputs, max_new_tokens):
        # inputs: (Batch, Seq_Len)
        for _ in range(max_new_tokens):
            # Crop to context length if needed
            cond_inputs = inputs[:, -self.context_length:]
            
            logits, _ = self(cond_inputs)
            # Take last token logits
            logits = logits[:, -1, :] 
            probs = torch.softmax(logits, dim=1)            
            
            idx_next = torch.multinomial(probs, num_samples=1) 
            inputs = torch.cat([inputs, idx_next], dim=1)
            
        return inputs