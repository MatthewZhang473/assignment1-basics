"""
ROTARY POSITIONAL EMBEDDING (RoPE) MATHEMATICAL EXPLANATION

RoPE applies a rotation to pairs of dimensions in the hidden representation.
For a 2D vector [x1, x2], a rotation by angle θ is defined by the matrix:

    [ x1' ] = [ cosθ  -sinθ ] [ x1 ] = [ x1*cosθ - x2*sinθ ]
    [ x2' ]   [ sinθ   cosθ ] [ x2 ]   [ x1*sinθ + x2*cosθ ]

The code optimizes this by grouping the terms:
    Result = (Original Vector * cosθ) + (Rotated Basis * sinθ)

Let x = [x1, x2].
Let rotate_adjacent(x) = [-x2, x1].

Then the rotation formula becomes:
    [x1', x2'] = [x1, x2] * cosθ + [-x2, x1] * sinθ
               = [x1*cosθ - x2*sinθ, x2*cosθ + x1*sinθ]

In high-dimensional space (d_k), we treat the vector as d_k/2 independent pairs.
Each pair (j) at token position (m) is rotated by an angle θ_m,j calculated as:
    θ_m,j = m * (theta ** (-2j / d_k))

Where:
    - m is the token position (0, 1, 2, ...)
    - j is the dimension index
    - theta is the base (typically 10,000)

The code performs this across all pairs simultaneously using element-wise
multiplication and the `rotate_adjacent` helper to swap and negate values.
"""

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
