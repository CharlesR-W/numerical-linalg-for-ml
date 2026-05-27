import math

import torch

from solutions._03_estimation import (
    hutchinson_trace,
    principal_angle,
    slq_density,
    weyl_bound,
)


def _random_symmetric(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A + A.T


def test_hutchinson_unbiased_large_m():
    """Hutchinson estimate should be within ~3 standard deviations of the truth.

    For Rademacher probes, Var(zᵀAz) = 2(‖A‖_F² − ‖diag(A)‖²) ≤ 2‖A‖_F².
    So Var of the m-sample mean is at most 2‖A‖_F²/m.
    """
    n = 30
    A = _random_symmetric(n)
    true_trace = torch.diagonal(A).sum().item()

    m = 2000
    est, _ = hutchinson_trace(lambda v: A @ v, n=n, m=m, probe_type="rademacher", seed=0)
    sigma_bound = math.sqrt(2 * (A**2).sum().item() / m)
    assert abs(est - true_trace) < 3 * sigma_bound, (
        f"|est - true| = {abs(est - true_trace):.3f} > 3σ = {3 * sigma_bound:.3f}"
    )


def test_hutchinson_rademacher_lower_variance_than_gaussian():
    """For matrices with diagonal mass, Rademacher should have lower variance."""
    n = 40
    A = _random_symmetric(n)
    # Add diagonal mass to make Rademacher's advantage manifest.
    A = A + 5.0 * torch.eye(n)

    _, rad_samples = hutchinson_trace(lambda v: A @ v, n=n, m=500,
                                       probe_type="rademacher", seed=0)
    _, gauss_samples = hutchinson_trace(lambda v: A @ v, n=n, m=500,
                                         probe_type="gaussian", seed=0)
    rad_var = torch.tensor(rad_samples).var().item()
    gauss_var = torch.tensor(gauss_samples).var().item()
    assert rad_var < gauss_var, (
        f"expected rad_var < gauss_var; got rad={rad_var:.2f}, gauss={gauss_var:.2f}"
    )


def test_slq_density_integrates_to_n():
    """∫ ρ(λ) dλ ≈ n / sigma·sqrt(2π) · sum(weights) → should be ≈ n by construction."""
    n = 30
    A = _random_symmetric(n)
    grid = torch.linspace(-15.0, 15.0, 600)
    density = slq_density(lambda v: A @ v, n=n, m_probes=20, s_lanczos=30,
                          grid=grid, sigma=0.3, seed=0)
    integral = torch.trapezoid(density, grid).item()
    assert abs(integral - n) < 0.2 * n, f"integral = {integral:.2f}, expected ~{n}"


def test_slq_density_peaks_near_eigenvalues():
    """SLQ density should be supported near the true eigenvalues."""
    n = 20
    A = torch.diag(torch.linspace(-3.0, 3.0, n).double()).float()
    true_eigs = torch.linalg.eigvalsh(A)
    grid = torch.linspace(-5.0, 5.0, 500)
    density = slq_density(lambda v: A @ v, n=n, m_probes=30, s_lanczos=20,
                          grid=grid, sigma=0.2, seed=0)
    # Find peak — should be inside the spectrum's support.
    peak_loc = grid[density.argmax()].item()
    assert true_eigs.min().item() - 0.5 <= peak_loc <= true_eigs.max().item() + 0.5, (
        f"density peaked at {peak_loc:.2f}, eigenvalues are in "
        f"[{true_eigs.min():.2f}, {true_eigs.max():.2f}]"
    )


def test_weyl_bound_holds():
    """For symmetric A + αE, |Δλ_k| ≤ α‖E‖_2."""
    n = 20
    torch.manual_seed(0)
    A = _random_symmetric(n)
    E = _random_symmetric(n, seed=1)
    E_norm = torch.linalg.matrix_norm(E, ord=2).item()

    A_eigs = torch.linalg.eigvalsh(A)
    for alpha in [0.01, 0.1, 1.0]:
        P_eigs = torch.linalg.eigvalsh(A + alpha * E)
        delta = weyl_bound(A_eigs, P_eigs)
        assert delta.max().item() <= alpha * E_norm + 1e-5, (
            f"Weyl violated at alpha={alpha}: max |Δλ|={delta.max():.4f}, "
            f"bound={alpha * E_norm:.4f}"
        )


def test_principal_angle_zero_for_same_subspace():
    n = 10
    torch.manual_seed(0)
    U = torch.linalg.qr(torch.randn(n, 3))[0]
    angle = principal_angle(U, U)
    assert angle < 1e-5


def test_principal_angle_perpendicular_subspaces():
    n = 10
    e = torch.eye(n)
    U = e[:, :3]
    V = e[:, 3:6]
    angle = principal_angle(U, V)
    assert abs(angle - math.pi / 2) < 1e-5
