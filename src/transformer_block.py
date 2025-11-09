from torch import nn
import torch
from src.rmsnorm import RMSNorm
from src.multihead_self_attention import CausalMultiHeadSelfAttention
from src.positionwise_ffw import SwiGLU


class TransformerBlock(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = None,
        theta: float = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.rms1 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.rms2 = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            device=device,
            dtype=dtype,
        )
        self.ffwd = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(self, x):

        x = x + self.attn(self.rms1(x))
        x = x + self.ffwd(self.rms2(x))
        return x
