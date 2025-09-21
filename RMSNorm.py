import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ---------------- RMSNorm ---------------- #
class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        # x: [B, T, C]
        norm = x.norm(2, dim=-1, keepdim=True)  # L2 norm over last dim
        rms = norm * (1.0 / math.sqrt(x.size(-1)))  # divide by sqrt(dim)
        return self.weight * (x / (rms + self.eps))
