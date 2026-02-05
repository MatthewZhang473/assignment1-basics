from jaxtyping import Float, Int
from torch import Tensor
from einops import reduce, rearrange
import torch


def cross_entropy(
    logits: Float[Tensor, "batch num_classes"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, ""]:

    max_logits = logits.max(dim=-1, keepdim=True).values  # (B, 1)
    log_sum_exp = (logits - max_logits).exp().sum(
        dim=-1, keepdim=True
    ).log() + max_logits  # (B,1)
    batch_indices = torch.arange(logits.shape[0])  # (B,)
    target_logits = logits[batch_indices, targets]  # (B,)

    per_example_loss = log_sum_exp.squeeze(dim=-1) - target_logits  # (B,)

    return per_example_loss.mean(dim=0, keepdim=False)  # ()


def cross_entropy_einops(
    logits: Float[Tensor, "batch num_classes"],
    targets: Int[Tensor, "batch"],
) -> Float[Tensor, ""]:

    max_logits = reduce(logits, "b v -> b 1", "max")
    lse = (logits - max_logits).exp()
    lse = reduce(lse, "b v -> b 1", "sum").log() + max_logits
    lse = rearrange(lse, "b 1 -> b")

    batch_indices = torch.arange(logits.shape[0])
    target_logits = logits[batch_indices, targets]

    return reduce(lse - target_logits, "b -> ", "mean")
