import numpy as np
import torch


def get_batch(x, batch_size, context_length, device):

    max_idx = len(x) - context_length
    ix = np.random.randint(0, max_idx, size=(batch_size,))

    x_batch = np.stack([x[i : i + context_length] for i in ix])
    y_batch = np.stack([x[i + 1 : i + context_length + 1] for i in ix])

    x_tensor = torch.from_numpy(x_batch.astype(np.int64)).to(device)
    y_tensor = torch.from_numpy(y_batch.astype(np.int64)).to(device)

    return x_tensor, y_tensor
