import torch

from solutions._01_krylov import lanczos


def _random_symmetric(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A + A.T


def test_lanczos_no_reorth_runs():
    A = _random_symmetric(40)
    ritz, Q = lanczos(lambda v: A @ v, dim=40, k=20, reorth="none")
    assert ritz.shape == (20,)
    assert Q.shape == (40, 20)


def test_lanczos_full_reorth_orthonormal():
    A = _random_symmetric(40)
    ritz, Q = lanczos(lambda v: A @ v, dim=40, k=20, reorth="full")
    QTQ = Q.T @ Q
    err = (QTQ - torch.eye(20)).abs().max().item()
    assert err < 1e-5, f"Q not orthonormal: max |QᵀQ - I| = {err:.2e}"


def test_lanczos_full_reorth_top_eigs():
    A = _random_symmetric(60, seed=42)
    true_eigs = torch.linalg.eigvalsh(A)
    top3 = true_eigs[-3:].flip(0)
    ritz, _ = lanczos(lambda v: A @ v, dim=60, k=30, reorth="full")
    top3_ritz = ritz.sort(descending=True).values[:3]
    assert torch.allclose(top3_ritz, top3, atol=1e-4)


def test_lanczos_selective_orthogonality():
    """Selective reorth should keep ||QᵀQ - I|| below 1e-3 over 30 steps."""
    A = _random_symmetric(40)
    _, Q = lanczos(lambda v: A @ v, dim=40, k=30, reorth="selective")
    QTQ = Q.T @ Q
    err = (QTQ - torch.eye(30)).abs().max().item()
    assert err < 1e-3, f"selective reorth failed: |QᵀQ-I|={err:.2e}"


def test_lanczos_no_reorth_loses_orthogonality_eventually():
    """Without reorth, we lose orthogonality by step ~25 on a hard matrix."""
    A = _random_symmetric(40, seed=7).float()
    _, Q = lanczos(lambda v: A @ v, dim=40, k=35, reorth="none")
    QTQ = Q.T @ Q
    err = (QTQ - torch.eye(35)).abs().max().item()
    assert err > 1e-3, f"expected orthogonality loss but err={err:.2e}"
