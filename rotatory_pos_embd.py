import torch
def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, nh, T, hd]
    def rotate_half(x):
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((-x2, x1), dim=-1).reshape_as(x)

    q_rot = (q * cos) + (rotate_half(q) * sin)
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot