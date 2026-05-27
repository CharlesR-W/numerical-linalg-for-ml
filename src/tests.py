"""In-notebook test helpers.

Each `test_*` function takes the reader's implementation, calls a reference
implementation from `solutions/`, and asserts agreement.  Designed to be called
inline in notebook cells: `tests.test_hvp(your_hvp)`.
"""
import torch

from src.tiny_models import toy_mlp


def _toy_loss(model, X, y):
    return torch.nn.functional.cross_entropy(model(X), y)


def test_hvp(your_hvp) -> None:
    """Verify a reader's HVP implementation against the reference.

    your_hvp signature: your_hvp(model, X, y, v) -> Tensor of shape v.shape
    """
    from solutions import _01_krylov as sol
    torch.manual_seed(0)
    model = toy_mlp(seed=1)
    X = torch.randn(8, 20)
    y = torch.randint(0, 4, (8,))
    P = sum(p.numel() for p in model.parameters())
    v = torch.randn(P)
    expected = sol.reference_hvp(model, X, y, v)
    got = your_hvp(model, X, y, v)
    assert got.shape == expected.shape, f"shape {got.shape} != {expected.shape}"
    assert torch.allclose(got, expected, atol=1e-5), (
        f"HVP mismatch (max |Δ|={(got - expected).abs().max():.2e})"
    )
    print(f"✓ HVP matches reference (max |Δ|={(got - expected).abs().max():.2e})")


def test_power_iteration(your_power) -> None:
    """Verify reader's power iteration on a diagonal matrix with a clear gap.

    your_power signature: your_power(matvec, dim, num_iters, seed) -> (eigval, eigvec)
    """
    torch.manual_seed(0)
    # Diagonal SPD with a clear gap (10 vs 3) so 200 iterations converge well.
    diag = torch.cat([torch.tensor([10.0, 3.0]), torch.rand(38) * 2.0])
    Q = torch.linalg.qr(torch.randn(40, 40))[0]
    A = Q @ torch.diag(diag) @ Q.T
    top_true = 10.0

    def matvec(v):
        return A @ v

    eigval, eigvec = your_power(matvec, dim=40, num_iters=200, seed=0)
    assert abs(eigval - top_true) < 1e-3, (
        f"power iter got {eigval:.4f}, true top eig {top_true:.4f}"
    )
    residual = (matvec(eigvec) - eigval * eigvec).norm().item()
    assert residual < 1e-3, f"residual = {residual:.2e} too large"
    print(f"✓ power iteration: top λ={eigval:.4f}, true={top_true:.4f}, "
          f"residual={residual:.2e}")


def test_lanczos(your_lanczos) -> None:
    """Verify Ritz values converge to extremes of a known matrix.

    your_lanczos signature:
        your_lanczos(matvec, dim, k, reorth='selective', seed=0) -> (ritz_vals, Q)
    """
    from solutions import _01_krylov as sol
    torch.manual_seed(0)
    A = torch.randn(60, 60)
    A = A + A.T
    eigs = torch.linalg.eigvalsh(A)
    top5 = eigs[-5:].flip(0)

    def matvec(v):
        return A @ v

    ritz, Q = your_lanczos(matvec, dim=60, k=30, reorth="selective", seed=0)
    ritz_sorted = ritz.sort(descending=True).values
    top5_ritz = ritz_sorted[:5]
    err = (top5_ritz - top5).abs().max().item()
    # Selective reorth doesn't reach full-reorth precision; 5e-2 is a generous
    # but still-meaningful sanity bound for top-5 Ritz convergence.
    assert err < 5e-2, f"top-5 Ritz off by max |Δ|={err:.2e}"
    QTQ = Q.T @ Q
    orth = (QTQ - torch.eye(Q.shape[1])).abs().max().item()
    assert orth < 1e-3, f"Q lost orthogonality: max |QᵀQ - I| = {orth:.2e}"
    print(f"✓ Lanczos: top-5 Ritz match true (max |Δ|={err:.2e}), "
          f"|QᵀQ-I|={orth:.2e}")
