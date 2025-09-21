import torch
import torch.nn as nn
from TransformerBlock import TransformerBlock
from RMSNorm import RMSNorm
class Transformer(nn.Module):
    def __init__(self, vocab_size, embd_dim, seq_len, num_heads, num_layers):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, embd_dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(embd_dim, num_heads, seq_len) for _ in range(num_layers)
        ])
        self.norm = RMSNorm(embd_dim)
        self.head = nn.Linear(embd_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        B, T = idx.shape
        x = self.token_emb(idx)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        logits = self.head(x)
        return logits
    