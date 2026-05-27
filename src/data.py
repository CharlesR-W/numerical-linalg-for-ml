"""Tiny cached MNIST loader downsampled to 7x7 = 49 features.

Caches to data/mnist_7x7_n{N}.pt to avoid re-downloading on every notebook run.
"""
from pathlib import Path

import torch
import torch.nn.functional as F

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download_full_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    """One-time torchvision pull; cached as full_mnist.pt."""
    full_cache = DATA_DIR / "full_mnist.pt"
    if full_cache.exists():
        blob = torch.load(full_cache)
        return blob["X"], blob["y"]
    from torchvision import datasets, transforms
    ds = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    X = torch.stack([ds[i][0] for i in range(len(ds))])  # (60000, 1, 28, 28)
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    torch.save({"X": X, "y": y}, full_cache)
    return X, y


def load_mnist_7x7(n: int = 500, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, y) with X.shape=(n, 49), y.shape=(n,).  Normalized to [0,1]-ish."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"mnist_7x7_n{n}_seed{seed}.pt"
    if cache.exists():
        blob = torch.load(cache)
        return blob["X"], blob["y"]
    X_full, y_full = _download_full_mnist()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X_full), generator=g)[:n]
    X = X_full[idx]
    y = y_full[idx]
    X = F.avg_pool2d(X, kernel_size=4).reshape(n, 49)
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-6)
    torch.save({"X": X, "y": y}, cache)
    return X, y
