import pytest

torch = pytest.importorskip("torch")

from .data_utils import generate_toy_data


def test_generate_toy_data_shapes():
    X, T, Y = generate_toy_data(n=50, p=4, seed=1)
    assert X.shape == (50, 4)
    assert T.shape == (50,)
    assert Y.shape == (50,)
