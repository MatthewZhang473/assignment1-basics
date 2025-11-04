from src.scaled_dot_product_attention import scaled_dot_product_attention
from src.rope import RotaryPositionalEmbedding
from src.linear import Linear
import torch
from torch import nn


class CausalMultiHeadSelfAttention(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 2048,
        theta: float = 1e-3,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_k = d_model / num_heads
        self.num_heads = num_heads
        self.rope = RotaryPositionalEmbedding(
            theta=theta, d_k=self.d_k, max_seq_len=max_seq_len, device=device
        )
        self.q_proj = Linear(
            in_features=d_model,
            out_features=num_heads * self.d_k,  # which is same as d_model -> d_model
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=num_heads * self.d_k,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=num_heads * self.d_k,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        q = self.q_proj(x).reshape(...)
        k = self.q_proj(x).reshape(...)
        v = self.q_proj(x).reshape(...)

        mask = ...

        return scaled_dot_product_attention(q, k, v, mask)
