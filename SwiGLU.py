import  torch 
import torch.nn as nn
import torch.nn.functional as F
class SwiGLUFFN(nn.Module):
    def __init__(self, embd_dim, ff_mult=2.66):
        super().__init__()
        hidden_dim = int(ff_mult * embd_dim)
        self.w1 = nn.Linear(embd_dim, hidden_dim, bias=False)  # linear for gating
        self.w2 = nn.Linear(embd_dim, hidden_dim, bias=False)  # linear for SiLU
        self.proj = nn.Linear(hidden_dim, embd_dim, bias=False)

    def forward(self, x):
        return self.proj(self.w1(x) * F.silu(self.w2(x)))