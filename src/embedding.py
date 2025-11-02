from torch import nn
from src.linear import Linear
import torch


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.embd_mat = torch.empty(
            (num_embeddings, embedding_dim), device=device, dtype=dtype
        )
        std = (1.0 / embedding_dim) ** 0.5
        nn.init.trunc_normal_(self.embd_mat, mean=0, std=std, a=-3, b=3)
        self.embd_mat = nn.Parameter(self.embd_mat)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embd_mat[token_ids]
