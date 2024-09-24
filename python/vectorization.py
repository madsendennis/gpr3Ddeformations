import numpy as np
import torch
from typing import Union


def vectorize(data: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(data, np.ndarray):
        return data.flatten()
    elif isinstance(data, torch.Tensor):
        return data.view(-1)
    else:
        raise TypeError("Input must be either a numpy array or a PyTorch tensor")


def unvectorize(
    data: Union[np.ndarray, torch.Tensor], dim: int
) -> Union[np.ndarray, torch.Tensor]:
    if isinstance(data, np.ndarray):
        n = int(len(data) / dim)
        return data.reshape((n, dim))
    elif isinstance(data, torch.Tensor):
        n = int(data.numel() / dim)
        return data.view(n, dim)
    else:
        raise TypeError("Input must be either a numpy array or a PyTorch tensor")
