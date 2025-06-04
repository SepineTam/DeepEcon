import pytest
torch = pytest.importorskip("torch")
linear = pytest.importorskip("sklearn.linear_model")
ensemble = pytest.importorskip("sklearn.ensemble")
from deepecon.dml import _default_ml


def test_default_model_y_output_shape():
    model = _default_ml.default_model_y(3)
    x = torch.randn(5, 3)
    y = model(x)
    assert y.shape == (5, 1)


def test_default_model_t_discrete_and_continuous():
    model_d = _default_ml.default_model_t(4, discrete=True)
    model_c = _default_ml.default_model_t(4, discrete=False)
    x = torch.randn(2, 4)
    assert model_d(x).shape == (2, 1)
    assert model_c(x).shape == (2, 1)


def test_default_sklearn_lasso():
    factory = _default_ml.default_sklearn_lasso(alpha=0.5)
    model = factory()
    assert isinstance(model, linear.LassoCV)
    assert 0.5 in model.alphas


def test_default_sklearn_ridge():
    factory = _default_ml.default_sklearn_ridge(alphas=[0.1, 1.0])
    model = factory()
    assert isinstance(model, linear.RidgeCV)
    assert model.alphas.tolist() == [0.1, 1.0]


def test_default_sklearn_forest():
    f_reg = _default_ml.default_sklearn_forest(discrete=False)
    f_clf = _default_ml.default_sklearn_forest(discrete=True)
    assert isinstance(f_reg(), ensemble.RandomForestRegressor)
    assert isinstance(f_clf(), ensemble.RandomForestClassifier)
