import torch


def clip_grad_norm(params, maximum_l2_norm, epsilon=1e-6):
    # 1. Convert to list so we can iterate twice
    params = [p for p in params if p.grad is not None]

    # 2. Calculate norm using PyTorch ops
    total_norm = torch.sqrt(sum(p.grad.detach().data.norm(2) ** 2 for p in params))

    # 3. Apply the scaling factor in-place
    if total_norm > maximum_l2_norm:
        clip_coef = maximum_l2_norm / (total_norm + epsilon)
        for p in params:
            p.grad.detach().mul_(clip_coef)
