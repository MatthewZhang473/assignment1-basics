import torch
from torch import nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """Construct a linear transformation module"""
        super().__init__()
        W = torch.empty((out_features, in_features), device=device, dtype=dtype)
        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(W, mean=0, std=std, a=-3, b=3)
        self.W = nn.Parameter(W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.einsum("b s i, o i -> b s o", x, self.W)
        return y
