"""Reference implementations for Notebook 3 (Hutchinson, SLQ, perturbation).

Estimation of spectral functionals (trace, density of states) for matrices
we can only access via matvec.  Plus eigenvalue perturbation tools.
"""
import math

import torch

from solutions._01_krylov import lanczos


def hutchinson_trace(matvec, n, m, probe_type="rademacher", seed=0):
    """Hutchinson trace estimator.

    For E[zzᵀ] = I, we have tr(A) = E[zᵀAz].  Average over m probes.

    Args:
        matvec: callable v -> Av for symmetric A.
        n: ambient dimension.
        m: number of random probes.
        probe_type: 'rademacher' (±1 uniform) or 'gaussian' (N(0,1)).
        seed: RNG seed.

    Returns: (estimate, list of per-probe zᵀAz values).
    """
    g = torch.Generator().manual_seed(seed)
    samples = []
    for _ in range(m):
        if probe_type == "rademacher":
            z = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1)
        elif probe_type == "gaussian":
            z = torch.randn(n, generator=g)
        else:
            raise ValueError(probe_type)
        Az = matvec(z)
        samples.append((z @ Az).item())
    return sum(samples) / len(samples), samples


def slq_density(matvec, n, m_probes, s_lanczos, grid, sigma, seed=0):
    """Stochastic Lanczos Quadrature for the spectral density of states.

    For each probe z, run s-step Lanczos starting at z.  The Ritz values are
    a discrete approximation to the spectrum, weighted by (first component of
    the corresponding eigenvector of T)^2.  Sum and smooth with a Gaussian.

    Args:
        matvec: callable v -> Av for symmetric A.
        n: ambient dimension.
        m_probes: number of stochastic probes (m in the literature).
        s_lanczos: Lanczos steps per probe (s in the literature).
        grid: 1D tensor of points where the density is evaluated.
        sigma: Gaussian smoothing width.
        seed: RNG seed.

    Returns: density values on `grid`, normalized so tr(A) ≈ n · ∫ ρ(λ) λ dλ
    in expectation.
    """
    g = torch.Generator().manual_seed(seed)
    density = torch.zeros_like(grid)

    for probe_idx in range(m_probes):
        z = (torch.randint(0, 2, (n,), generator=g).float() * 2 - 1)
        z = z / z.norm()

        # Run Lanczos from this probe; need T tridiagonal + first-component
        # weights of T's eigenvectors.
        alphas, betas = _lanczos_tridiag(matvec, z, s_lanczos)
        T = torch.diag(alphas) + torch.diag(betas, 1) + torch.diag(betas, -1)
        eigvals_T, eigvecs_T = torch.linalg.eigh(T)
        weights = eigvecs_T[0, :] ** 2  # (s_lanczos,)

        for theta, w in zip(eigvals_T.tolist(), weights.tolist()):
            density += w * torch.exp(-((grid - theta) ** 2) / (2 * sigma ** 2))

    density = density * n / m_probes / (sigma * math.sqrt(2 * math.pi))
    return density


def _lanczos_tridiag(matvec, q0, k):
    """Lanczos with full reorth, returning just the tridiagonal (alpha, beta)."""
    n = q0.numel()
    q = q0 / q0.norm()
    Q = torch.zeros(n, k)
    alphas = torch.zeros(k)
    betas = torch.zeros(k - 1)
    q_prev = torch.zeros(n)
    beta_prev = 0.0

    for j in range(k):
        Q[:, j] = q
        Aq = matvec(q)
        alpha = q @ Aq
        alphas[j] = alpha
        r = Aq - alpha * q - beta_prev * q_prev
        if j > 0:
            r = r - Q[:, : j + 1] @ (Q[:, : j + 1].T @ r)
            r = r - Q[:, : j + 1] @ (Q[:, : j + 1].T @ r)
        beta = r.norm().item()
        if j < k - 1:
            betas[j] = beta
            if beta < 1e-14:
                break
            q_prev = q
            q = r / beta
            beta_prev = beta
    return alphas, betas


def weyl_bound(A_eigs, perturbed_eigs):
    """For symmetric A and A+E, |λ_k(A+E) - λ_k(A)| ≤ ‖E‖_2 (sorted matching)."""
    a_sorted = torch.sort(A_eigs).values
    p_sorted = torch.sort(perturbed_eigs).values
    return (a_sorted - p_sorted).abs()


def principal_angle(U, V):
    """Largest principal angle between two subspaces (columns of U and V).

    Both U and V should have orthonormal columns; otherwise we orthonormalize.
    Returns the angle in radians (in [0, pi/2]).
    """
    if U.shape[1] != V.shape[1]:
        raise ValueError("Subspaces must have same dimension")
    Qu, _ = torch.linalg.qr(U)
    Qv, _ = torch.linalg.qr(V)
    s = torch.linalg.svdvals(Qu.T @ Qv)
    s = s.clamp(-1.0, 1.0)
    cos_min = s.min().item()
    return math.acos(cos_min)
