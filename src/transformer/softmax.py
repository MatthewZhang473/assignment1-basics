import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Applying softmax to the ith dimension"""
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = (x - x_max).exp()
    return x_exp / x_exp.sum(dim=dim, keepdim=True)


if __name__ == "__main__":
    x = torch.tensor([[1.0, 2.0, 3.0]])
    print(softmax(x, dim=-1))
