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
        self.d_k = d_model // num_heads
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
        self.o_proj = Linear(
            in_features=num_heads * self.d_k,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor, token_ids: torch.Tensor = None) -> torch.Tensor:

        batch_size, seq_len, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        def split_heads(t):
            # (B, S, H*D) -> (B, H, S, D)
            t = t.view(batch_size, seq_len, self.num_heads, self.d_k)
            return t.transpose(1, 2)

        q = split_heads(q)  # shape: (B, H, S, d_k)
        k = split_heads(k)
        v = split_heads(v)

        if token_ids is None:
            token_ids = torch.arange(0, seq_len, device=x.device)

        q = self.rope(q, token_ids)
        k = self.rope(k, token_ids)
        mask = torch.ones((seq_len, seq_len), dtype=torch.bool, device=x.device)
        mask = torch.tril(mask)

        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous()
        out = out.view(batch_size, seq_len, self.num_heads * self.d_k)
        return self.o_proj(out)
