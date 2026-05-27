import torch

from solutions._02_randomized import (
    entk_explicit,
    entk_matvec,
    randomized_eigh,
    stable_rank,
)
from src.tiny_models import toy_mlp


def _random_symmetric_with_decay(n, decay_type, seed=0):
    """Build a symmetric n×n matrix with a chosen spectral decay."""
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(n, n, generator=g))[0]
    if decay_type == "fast":
        eigs = 2.0 ** (-torch.arange(n).float())
    elif decay_type == "slow":
        eigs = 1.0 / (1.0 + torch.arange(n).float())
    elif decay_type == "flat":
        eigs = torch.ones(n)
    else:
        raise ValueError(decay_type)
    return Q @ torch.diag(eigs) @ Q.T


def test_stable_rank_one_dominant():
    """Rank-1 matrix should have stable rank 1."""
    u = torch.randn(50)
    A = u[:, None] @ u[None, :]
    assert abs(stable_rank(A) - 1.0) < 1e-3


def test_stable_rank_isotropic():
    """Eigvals all equal to 1: stable rank = n."""
    n = 30
    A = _random_symmetric_with_decay(n, "flat")
    assert abs(stable_rank(A) - n) < 0.5


def test_randomized_eigh_fast_decay():
    """On fast-decay matrix, rSVD with small k should be very accurate."""
    n = 100
    A = _random_symmetric_with_decay(n, "fast")
    true_eigs = torch.linalg.eigvalsh(A).abs().sort(descending=True).values

    eigvals, _ = randomized_eigh(lambda v: A @ v, n=n, k=5, oversample=10, seed=0)
    rel_err = (eigvals.abs() - true_eigs[:5]).abs() / true_eigs[:5]
    assert rel_err.max().item() < 1e-4, f"rel err {rel_err.max():.2e}"


def test_randomized_eigh_eigenvectors_eigvalue_relation():
    n = 60
    A = _random_symmetric_with_decay(n, "fast")
    eigvals, eigvecs = randomized_eigh(lambda v: A @ v, n=n, k=3, oversample=10, seed=0)
    for i in range(3):
        residual = (A @ eigvecs[:, i] - eigvals[i] * eigvecs[:, i]).norm().item()
        assert residual < 5e-4, f"residual={residual:.2e} at i={i}"


def test_randomized_eigh_power_iterations_help_slow_decay():
    """With slow decay, power iterations should reduce error vs no power."""
    n = 100
    A = _random_symmetric_with_decay(n, "slow")
    true_top = torch.linalg.eigvalsh(A).abs().sort(descending=True).values[:5]

    eigs_no_power, _ = randomized_eigh(lambda v: A @ v, n=n, k=5, oversample=5, n_power=0, seed=0)
    eigs_2_power, _ = randomized_eigh(lambda v: A @ v, n=n, k=5, oversample=5, n_power=2, seed=0)

    err_no = (eigs_no_power.abs() - true_top).abs().mean().item()
    err_p2 = (eigs_2_power.abs() - true_top).abs().mean().item()
    assert err_p2 <= err_no + 1e-6, (
        f"power iter should help: no_power err={err_no:.2e}, "
        f"2_power err={err_p2:.2e}"
    )


def test_entk_matvec_matches_explicit():
    """Matrix-free eNTK matvec should match K @ v for the explicit K."""
    torch.manual_seed(0)
    model = toy_mlp(seed=1)
    X = torch.randn(12, 20)
    v = torch.randn(12)

    K = entk_explicit(model, X)
    expected = K @ v
    got = entk_matvec(model, X, v)
    assert torch.allclose(got, expected, atol=1e-4), (
        f"max |Δ| = {(got - expected).abs().max():.2e}"
    )


def test_entk_explicit_psd():
    """eNTK Gram is symmetric PSD by construction."""
    torch.manual_seed(0)
    model = toy_mlp(seed=2)
    X = torch.randn(15, 20)
    K = entk_explicit(model, X)
    assert torch.allclose(K, K.T, atol=1e-5)
    eigs = torch.linalg.eigvalsh(K)
    assert eigs.min().item() > -1e-5, f"min eig = {eigs.min():.2e} (not PSD)"
