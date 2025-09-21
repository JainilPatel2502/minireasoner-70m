from MultiheadAttention import MultiHeadAttention
from SwiGLU import SwiGLUFFN
from RMSNorm import RMSNorm
import torch.nn  as nn
class TransformerBlock(nn.Module):
    def __init__(self, embd_dim, num_heads, seq_len):
        super().__init__()
        self.attn = MultiHeadAttention(num_heads, embd_dim, seq_len)
        self.ff = SwiGLUFFN(embd_dim, ff_mult=2.66)  # ✅ SwiGLU FFN
        self.rms1 = RMSNorm(embd_dim)
        self.rms2 = RMSNorm(embd_dim)

    def forward(self, x):
        x = x + self.attn(self.rms1(x))
        x = x + self.ff(self.rms2(x))
        return x