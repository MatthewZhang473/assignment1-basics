from torch import nn
import torch
from src.embedding import Embedding
from src.rmsnorm import RMSNorm
from src.linear import Linear
from src.softmax import softmax
from src.transformer_block import TransformerBlock


class TransformerLM(nn.Module):

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        vocab_size: int,
        context_length: int,
        num_layers: int,
        theta: float = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.embd = Embedding(num_embeddings=vocab_size, embedding_dim=d_model)
        self.tbs = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=theta,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.rms = RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm = Linear(
            in_features=d_model, out_features=vocab_size, device=device, dtype=dtype
        )

    def forward(self, token_ids):

        x = self.embd(token_ids=token_ids)
        for tb in self.tbs:
            x = tb(x)
        x = self.rms(x)
        x = self.lm(x)
        x = softmax(x)
        return x
