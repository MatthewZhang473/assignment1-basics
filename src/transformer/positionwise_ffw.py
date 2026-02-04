import torch
from torch import nn
from src.linear import Linear
import math


def silu(x):
    return x * torch.sigmoid(x)


class GLU(nn.Module):

    def __init__(self, d_in, d_out, act_fn=silu, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_in, d_out, device, dtype)
        self.w3 = Linear(d_in, d_out, device, dtype)
        self.act_fn = act_fn

    def forward(self, x):
        return self.act_fn(self.w1(x)) * self.w3(x)


class SwiGLU(nn.Module):

    def __init__(self, d_model, d_ff=None, device=None, dtype=None):
        super().__init__()

        if d_ff is None:
            d_ff = math.ceil(d_model * 8 / 3 / 64) * 64
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.glu = GLU(d_model, d_ff, silu, device, dtype)

    def forward(self, x):
        return self.w2(self.glu(x))
