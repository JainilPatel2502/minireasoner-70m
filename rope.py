import torch
def build_rope_cache(seq_len, head_dim, device):
    theta = 10000.0 ** (-torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    pos = torch.arange(seq_len, device=device).float()
    freqs = torch.einsum("i,j->ij", pos, theta)  # [T, hd/2]
    emb = torch.cat((freqs, freqs), dim=-1)  # [T, hd]
    cos, sin = emb.cos()[None, None, :, :], emb.sin()[None, None, :, :]
    return cos, sin