import pytest
torch = pytest.importorskip("torch")
import numpy as np
import random

from deepecon.dml import _utils


def test_set_random_seed_determinism():
    _utils.set_random_seed(123)
    val1 = (random.random(), np.random.rand(), torch.randn(1).item())
    _utils.set_random_seed(123)
    val2 = (random.random(), np.random.rand(), torch.randn(1).item())
    assert val1 == val2


def test_set_random_seed_negative():
    with pytest.raises(ValueError):
        _utils.set_random_seed(-1)


def test_get_device():
    device = _utils.get_device()
    assert device in {"cuda", "cpu"}


def test_split_indices_basic():
    splits = _utils.split_indices(10, 5, shuffle=False)
    assert len(splits) == 5
    all_val = sum((val for _, val in splits), [])
    assert sorted(all_val) == list(range(10))
    for train, val in splits:
        assert set(train).isdisjoint(val)


def test_split_indices_invalid():
    with pytest.raises(ValueError):
        _utils.split_indices(5, 1)
    with pytest.raises(ValueError):
        _utils.split_indices(5, 6)


def test_compute_theta_from_residuals():
    t = torch.tensor([1.0, 1.0, 1.0])
    y = torch.tensor([2.0, 2.0, 2.0])
    theta = _utils.compute_theta_from_residuals(t, y)
    assert theta == pytest.approx(2.0)

    with pytest.raises(ValueError):
        _utils.compute_theta_from_residuals(torch.tensor([0.0, 0.0]), torch.tensor([1.0, 1.0]))

    with pytest.raises(ValueError):
        _utils.compute_theta_from_residuals(torch.tensor([1.0, 2.0]), torch.tensor([1.0]))


def test_bootstrap_ate_ci():
    t = torch.ones(20)
    y = torch.ones(20) * 2
    lower, upper = _utils.bootstrap_ate_ci(t, y, n_bootstrap=100, seed=0)
    assert lower < 2.0 < upper

    with pytest.raises(ValueError):
        _utils.bootstrap_ate_ci(t, y, n_bootstrap=0)
    with pytest.raises(ValueError):
        _utils.bootstrap_ate_ci(t, y, alpha=1.0)
