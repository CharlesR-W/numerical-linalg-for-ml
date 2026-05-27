"""Reference implementations for Notebook 2 (Randomized methods + eNTK).

These are the source of truth that the notebook's exercises check against.
"""
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jacrev, jvp


def stable_rank(A: torch.Tensor) -> float:
    """Stable / effective rank: ‖A‖_F² / ‖A‖_2².

    Small (≈1) when one singular value dominates.  ≈ rank(A) when all
    singular values are equal.  Predicts whether rSVD will help.
    """
    fro_sq = (A**2).sum().item()
    op_sq = torch.linalg.matrix_norm(A, ord=2).item() ** 2
    return fro_sq / op_sq


def randomized_eigh(matvec, n, k, oversample=10, n_power=0, seed=0):
    """Halko-Martinsson-Tropp range finder for SYMMETRIC matrices.

    Args:
        matvec: callable v -> Av for symmetric A.
        n: ambient dimension.
        k: number of eigenpairs to return.
        oversample: extra sketch columns beyond k.  Default 10.
        n_power: subspace power iterations to amplify dominant subspace.
        seed: RNG seed for the sketch.

    Returns: (eigvals, eigvecs) where eigvals.shape = (k,) sorted by descending
    magnitude, and eigvecs.shape = (n, k).
    """
    g = torch.Generator().manual_seed(seed)
    sketch_size = k + oversample

    Omega = torch.randn(n, sketch_size, generator=g)
    Y = torch.stack([matvec(Omega[:, i]) for i in range(sketch_size)], dim=1)

    for _ in range(n_power):
        Y, _ = torch.linalg.qr(Y)
        Y = torch.stack([matvec(Y[:, i]) for i in range(sketch_size)], dim=1)

    Q, _ = torch.linalg.qr(Y)

    AQ = torch.stack([matvec(Q[:, j]) for j in range(sketch_size)], dim=1)
    B = Q.T @ AQ
    B = (B + B.T) / 2

    eigvals_B, eigvecs_B = torch.linalg.eigh(B)
    idx = eigvals_B.abs().sort(descending=True).indices[:k]
    eigvals = eigvals_B[idx]
    eigvecs = Q @ eigvecs_B[:, idx]
    return eigvals, eigvecs


def entk_matvec(model, X, v):
    """Matrix-free matvec for the scalar empirical NTK Gram matrix.

    For a network with scalar output ``f(x) = model(x).sum(-1)``, the NTK is
    K[i, j] = <∇θ f(x_i), ∇θ f(x_j)>.  This computes K @ v without forming K:

        Step 1 (VJP): g = ∇θ Σⱼ vⱼ f(x_j)  ∈ R^P
        Step 2 (JVP): (Kv)_i = <∇θ f(x_i), g>  ∈ R

    Cost: one VJP + one JVP, regardless of N or P.

    Args:
        model: a torch.nn.Module.
        X: (N, D) input batch.
        v: (N,) tensor.

    Returns: (N,) tensor with K @ v.
    """
    params = {n: p.detach() for n, p in model.named_parameters()}

    def f_scalar(params_dict):
        return functional_call(model, params_dict, (X,)).sum(dim=-1)  # (N,)

    def weighted(params_dict):
        return (f_scalar(params_dict) * v).sum()

    g_dict = grad(weighted)(params)

    _, kv = jvp(f_scalar, (params,), (g_dict,))
    return kv


def entk_explicit(model, X):
    """Build the full N×N eNTK Gram matrix by materializing the Jacobian.

    Used only as ground truth for tests / for the matrix-free vs materialized
    timing comparison.  Cost: O(N·P) memory + N forward+backward sweeps.
    """
    params = {n: p.detach() for n, p in model.named_parameters()}

    def f_scalar(params_dict):
        return functional_call(model, params_dict, (X,)).sum(dim=-1)  # (N,)

    J_dict = jacrev(f_scalar)(params)  # dict of (N, *param_shape)
    J = torch.cat(
        [j.reshape(j.shape[0], -1) for j in J_dict.values()], dim=1
    )  # (N, P)
    return J @ J.T
