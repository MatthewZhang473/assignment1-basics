import torch
from torch import nn


def rotate_adjacent(x):
    x_even = x[..., 0::2]  # (arbitrary batching shape, sequence length, d_k)
    x_odd = x[..., 1::2]
    return torch.stack((-x_odd, x_even), dim=-1).reshape_as(x)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):

    return (x * cos) + rotate_adjacent(x) * sin


class RotaryPositionalEmbedding(nn.Module):

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        self.d_k = d_k  # dimension of query / key vectors
        self.max_seq_len = max_seq_len
        self.device = device
        self.pass_rope = False

        if theta == 0:
            self.pass_rope = True
        else:
            inv_freq = 1.0 / (
                theta
                ** (torch.arange(0, d_k, 2).to(device=device, dtype=torch.float) / d_k)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:

        if self.pass_rope:
            return x

        freqs = torch.einsum(
            "...n, m -> ...n m", token_positions, self.inv_freq
        )  # outer product - result in shape (..., sequence_length, d_k/2)

        cos = freqs.cos()
        sin = freqs.sin()
        # Broadcast to match the even/odd layout of x
        cos = torch.repeat_interleave(
            cos, repeats=2, dim=-1
        )  # [f0,f1,f2, ...] -> [f0,f0,f1,f1 ... fk,fk]
        sin = torch.repeat_interleave(sin, repeats=2, dim=-1)

        return apply_rotary_pos_emb(x, cos, sin)
