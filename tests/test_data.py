import torch
from src.data import load_mnist_7x7


def test_load_mnist_7x7_shape():
    X, y = load_mnist_7x7(n=200, seed=0)
    assert X.shape == (200, 49)
    assert y.shape == (200,)
    assert X.dtype == torch.float32
    assert y.dtype == torch.long


def test_load_mnist_7x7_deterministic():
    X1, y1 = load_mnist_7x7(n=50, seed=42)
    X2, y2 = load_mnist_7x7(n=50, seed=42)
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


def test_load_mnist_7x7_normalized():
    X, _ = load_mnist_7x7(n=100, seed=0)
    assert X.min() >= -1.0
    assert X.max() <= 4.0
