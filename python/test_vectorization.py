import torch
import numpy as np
from typing import Union

from vectorization import vectorize, unvectorize


def test_vectorize():
    input_data = torch.tensor([[1, 2, 3], [4, 5, 6]])
    expected_output = torch.tensor([1, 2, 3, 4, 5, 6])
    assert torch.all(vectorize(input_data).eq(expected_output))
    assert np.array_equal(vectorize(input_data).numpy(), expected_output.numpy())


def test_unvectorize():
    input_data = torch.tensor([1, 2, 3, 4, 5, 6])
    expected_output = torch.tensor([[1, 2, 3], [4, 5, 6]])
    assert torch.all(unvectorize(input_data, 3).eq(expected_output))
    assert np.array_equal(unvectorize(input_data.numpy(), 3), expected_output.numpy())


def vectorize_unvectorize_roundtrip(data: Union[np.ndarray, torch.Tensor]) -> bool:
    vectorized = vectorize(data)
    unvectorized = unvectorize(vectorized, data.shape[1])
    if isinstance(data, torch.Tensor):
        return torch.all(data.eq(unvectorized))
    else:
        return np.array_equal(data, unvectorized)


def test_vectorize_unvectorize_roundtrip():
    input_data = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    assert vectorize_unvectorize_roundtrip(input_data)
    assert vectorize_unvectorize_roundtrip(input_data.numpy())

    random_data = torch.rand(10, 5)
    assert vectorize_unvectorize_roundtrip(random_data)
    assert vectorize_unvectorize_roundtrip(random_data.numpy())
