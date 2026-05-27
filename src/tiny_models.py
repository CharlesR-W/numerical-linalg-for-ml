"""Tiny deterministic models used throughout the tutorial.

Three sizes:
- toy_mlp:  ~250 params.  Hessian materializable.  Ground-truth checks.
- tiny_mlp: ~2k params.  Hessian still materializable but tighter.  Workhorse.
- tiny_cnn: defined in a later phase for the capstone.
"""
import torch
import torch.nn as nn


def _seed_init(module: nn.Module, seed: int) -> None:
    """Seed PyTorch and reinitialize the module's parameters."""
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        if p.dim() >= 2:
            nn.init.kaiming_normal_(p, generator=g)
        else:
            nn.init.zeros_(p)


def toy_mlp(seed: int = 0) -> nn.Module:
    """20-dim input, width 8, depth 2, 4-class output.  ~250 params."""
    m = nn.Sequential(
        nn.Linear(20, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.Tanh(),
        nn.Linear(8, 4),
    )
    _seed_init(m, seed)
    return m


def tiny_mlp(seed: int = 0) -> nn.Module:
    """49-dim input (7x7 MNIST), width 32, depth 2, 10-class.  ~2k params."""
    m = nn.Sequential(
        nn.Linear(49, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 10),
    )
    _seed_init(m, seed)
    return m


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
