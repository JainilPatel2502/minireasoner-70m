import  torch 
import torch.nn as nn
import torch.nn.functional as F
from rope import build_rope_cache
from rotatory_pos_embd import apply_rotary_pos_emb
class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, embd_dim, seq_len):
        super().__init__()
        assert embd_dim % num_heads == 0, "Embedding dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embd_dim // num_heads
        self.seq_len = seq_len

        self.qkv = nn.Linear(embd_dim, 3 * embd_dim, bias=False)
        self.proj = nn.Linear(embd_dim, embd_dim, bias=False)

        # rope cache
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        if self.cos is None or self.cos.shape[2] < T:
            cos, sin = build_rope_cache(T, self.head_dim, x.device)
            self.cos, self.sin = cos, sin

        qkv = self.qkv(x).view(B, T, self.num_heads, 3 * self.head_dim)
        q, k, v = qkv.split(self.head_dim, dim=-1)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # apply RoPE
        cos, sin = self.cos[:, :, :T, :], self.sin[:, :, :T, :]
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(out)
