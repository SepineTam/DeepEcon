import pytest
torch = pytest.importorskip("torch")

from deepecon.dml.dml import DML


def simple_builder(dim):
    return torch.nn.Linear(dim, 1)


def test_dml_is_abstract():
    with pytest.raises(TypeError):
        DML(simple_builder, simple_builder)
