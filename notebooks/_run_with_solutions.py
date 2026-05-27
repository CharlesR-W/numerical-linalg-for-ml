"""Inject solutions into 01_krylov.ipynb and execute it.

Used as the CI smoke test for Notebook 1.  Run:
    uv run python notebooks/_run_with_solutions.py
"""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).parent / "01_krylov.ipynb"

# Imports + override functions.  Inserted into every cell that contains a
# `raise NotImplementedError`, after the def block ends and before the function
# is used downstream.
INJECT_IMPORTS_AND_OVERRIDES = """

# === INJECTED BY _run_with_solutions.py ===
from solutions._01_krylov import (
    hvp_double_backward as _ref_hvp_double,
    hvp_jvp_of_grad     as _ref_hvp_jvp,
    hvp_finite_difference as _ref_hvp_fd,
    power_iteration as _ref_power,
    power_iteration_deflated as _ref_power_defl,
    lanczos as _ref_lanczos,
)
import torch as _torch

def _shim_lanczos_no_reorth(matvec, dim, k, seed=0):
    ritz, Q = _ref_lanczos(matvec, dim=dim, k=k, reorth='none', seed=seed)
    return ritz, Q

def _shim_lanczos_track_orth(matvec, dim, k, reorth='none', seed=0):
    g = _torch.Generator().manual_seed(seed)
    q = _torch.randn(dim, generator=g); q = q / q.norm()
    Q = _torch.zeros(dim, k); alphas = _torch.zeros(k); betas = _torch.zeros(k-1)
    q_prev = _torch.zeros(dim); beta_prev = 0.0; orth_hist = []
    for j in range(k):
        Q[:, j] = q
        Aq = matvec(q)
        alpha = q @ Aq; alphas[j] = alpha
        r = Aq - alpha*q - beta_prev*q_prev
        if reorth == 'full' and j > 0:
            r = r - Q[:, :j+1] @ (Q[:, :j+1].T @ r)
            r = r - Q[:, :j+1] @ (Q[:, :j+1].T @ r)
        elif reorth == 'selective' and j > 0 and r.norm() < 0.717 * Aq.norm():
            r = r - Q[:, :j+1] @ (Q[:, :j+1].T @ r)
        beta = r.norm().item()
        if j < k-1:
            betas[j] = beta
            if beta < 1e-14: break
            q_prev = q; q = r / beta; beta_prev = beta
        orth_hist.append((Q[:, :j+1].T @ Q[:, :j+1] - _torch.eye(j+1)).abs().max().item())
    T = _torch.diag(alphas) + _torch.diag(betas, 1) + _torch.diag(betas, -1)
    return _torch.linalg.eigvalsh(T), Q, orth_hist

# Override stub functions with references.
hvp_double_backward = _ref_hvp_double
hvp_jvp_of_grad = _ref_hvp_jvp
hvp_finite_difference = _ref_hvp_fd
power_iteration = _ref_power
power_iteration_deflated = _ref_power_defl
lanczos_no_reorth = _shim_lanczos_no_reorth
lanczos_track_orth = _shim_lanczos_track_orth
# === END INJECTION ===

"""


def main():
    nb = nbformat.read(NB, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        src = cell.source

        # If this cell has a NotImplementedError stub, inject overrides
        # AFTER the def block ends (after `    raise NotImplementedError\n\n`).
        # This places overrides between the def and downstream uses.
        if "raise NotImplementedError" in src:
            # Match end of def body: indented raise followed by blank line.
            src = src.replace(
                "    raise NotImplementedError\n\n",
                "    raise NotImplementedError\n" + INJECT_IMPORTS_AND_OVERRIDES,
            )
            # Edge case: cell ends without a blank line after raise.
            if src.endswith("    raise NotImplementedError"):
                src = src + "\n" + INJECT_IMPORTS_AND_OVERRIDES

        # Section 0 inline "YOUR CODE HERE" patches.
        if "op_norm = None" in src:
            src = src.replace(
                "op_norm = None  # YOUR CODE HERE: compute ||A||_2",
                "op_norm = torch.linalg.matrix_norm(A, ord=2).item()",
            ).replace(
                "fro_norm = None  # YOUR CODE HERE: compute ||A||_F",
                "fro_norm = torch.linalg.matrix_norm(A, ord='fro').item()",
            )
        if "x_hat = None" in src:
            src = src.replace(
                "x_hat = None  # YOUR CODE HERE: solve A x_hat = b in fp32",
                "x_hat = torch.linalg.solve(A, b)",
            )
        if "eps32 = None" in src:
            src = src.replace(
                "eps32 = None  # YOUR CODE HERE",
                "eps32 = find_eps(torch.float32)",
            ).replace(
                "eps64 = None  # YOUR CODE HERE",
                "eps64 = find_eps(torch.float64)",
            )

        # Section 6 "ritz_init = None" / "ritz_trained = None" stubs.
        if "ritz_init = None" in src:
            src = src.replace(
                "ritz_init = None",
                "ritz_init, _, _ = lanczos_track_orth(matvec_init, "
                "dim=P_mnist, k=30, reorth='selective')",
            )
        if "ritz_trained = None" in src:
            src = src.replace(
                "ritz_trained = None  # YOUR CODE HERE: same as ritz_init but for the trained model.",
                "ritz_trained, _, _ = lanczos_track_orth(matvec_trained, "
                "dim=P_mnist, k=30, reorth='selective')",
            )

        cell.source = src

    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
    print(f"✓ notebook executed end-to-end ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
