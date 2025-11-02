import torch
from torch import nn
from src.softmax import softmax


def scaled_dot_product_attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """
    q, k: (..., seq_len, d_k)
    v:    (..., seq_len, d_v)
    returns: (..., seq_len, d_v)
    """
    d_k = q.shape[-1]
    scores = torch.einsum("... i d, ... j d-> ... i j", q, k) / d_k**0.5
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)

    return attn @ v


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 4
    d_k = 8
    d_v = 8

    q = torch.randn(batch_size, seq_len, d_k)
    k = torch.randn(batch_size, seq_len, d_k)
    v = torch.randn(batch_size, seq_len, d_v)

    # Optional mask (not used here)
    mask = None  # or torch.ones(batch_size, seq_len, seq_len)

    out = scaled_dot_product_attention(q, k, v, mask)
    print("q shape:", q.shape)
    print("k shape:", k.shape)
    print("v shape:", v.shape)
    print("output shape:", out.shape)
    print("output:\n", out)
