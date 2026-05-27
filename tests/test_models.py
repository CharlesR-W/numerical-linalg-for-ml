import torch
from src.tiny_models import toy_mlp, tiny_mlp, count_params


def test_toy_mlp_param_count():
    m = toy_mlp()
    assert 200 <= count_params(m) <= 350, "toy_mlp must stay ~250 params"


def test_tiny_mlp_param_count():
    m = tiny_mlp()
    assert 1500 <= count_params(m) <= 3000, "tiny_mlp ~2k params"


def test_toy_mlp_deterministic():
    a = toy_mlp(seed=0)
    b = toy_mlp(seed=0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_toy_mlp_forward_shape():
    m = toy_mlp()
    x = torch.randn(3, 20)
    assert m(x).shape == (3, 4)


def test_tiny_mlp_forward_shape():
    m = tiny_mlp()
    x = torch.randn(3, 49)
    assert m(x).shape == (3, 10)
