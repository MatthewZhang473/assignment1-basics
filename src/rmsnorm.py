from torch import nn
import torch


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.gain = torch.empty(d_model, device=device, dtype=dtype)  # (d_model)
        std = (1.0 / d_model) ** 0.5
        nn.init.trunc_normal_(self.gain, mean=0.0, std=std, a=-3.0, b=3.0)
        self.gain = nn.Parameter(self.gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process an input tensor of shape
        (batch_size, sequence_length, d_model)
        and return a tensor of the same shape."""
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        rms = (x.pow(2).mean(dim=-1, keepdim=True) + self.eps).sqrt()  # (batch, seq, 1)
        out = self.gain * x / rms  # (batch, seq, d_model)
        return out.to(orig_dtype)
