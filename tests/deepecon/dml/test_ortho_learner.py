import pytest
torch = pytest.importorskip("torch")

from deepecon.dml._ortho_learner import OrthogonalLearner


class DummyOrthogonalLearner(OrthogonalLearner):
    def fit(self, X, T, Y, W=None):
        self._theta = 1.0
        self._resid_y = torch.zeros(len(Y))
        self._resid_t = torch.zeros(len(T))

    def effect(self, X_test: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def simple_builder(dim):
    return torch.nn.Linear(dim, 1)


def test_orthogonal_learner_methods():
    learner = DummyOrthogonalLearner(simple_builder, simple_builder)

    with pytest.raises(RuntimeError):
        learner.ate()

    with pytest.raises(RuntimeError):
        learner.get_residuals()

    learner.fit(torch.randn(5, 3), torch.ones(5), torch.ones(5))
    assert learner.ate() == 1.0
    resid_y, resid_t = learner.get_residuals()
    assert resid_y.shape[0] == 5
    assert resid_t.shape[0] == 5

    with pytest.raises(NotImplementedError):
        learner.effect(torch.randn(1, 3))


def test_base_abstract_fit():
    base = OrthogonalLearner(simple_builder, simple_builder)
    with pytest.raises(NotImplementedError):
        base.fit(torch.randn(2, 2), torch.ones(2), torch.ones(2))
