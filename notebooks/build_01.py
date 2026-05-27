"""Build script for Notebook 1 (Krylov: power iteration + Lanczos).

Run: `uv run python notebooks/build_01.py`
Output: notebooks/01_krylov.ipynb

Each `add_*` helper appends a cell.  The body of build() is a linear
reading of the design doc's Notebook 1 sections.
"""
from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/01_krylov.ipynb"


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []
    _section_preamble(cells)
    _section_0_nla_prelim(cells)
    _section_1_hvp(cells)
    _section_2_power(cells)
    _section_3_deflation(cells)
    _section_4_lanczos(cells)
    _section_5_orthogonality(cells)
    _section_6_hessian_topk(cells)
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def _section_preamble(cells):
    cells.append(md("""
# Notebook 1 — Krylov methods for the Hessian

ARENA-style hands-on tutorial on matrix-free eigenvalue methods, with the
loss Hessian of a tiny neural network as the running example.

**Time:** ~145 min.  **Prerequisites:** PyTorch, basic autodiff.

**Sections:**
0. NLA you'll need (norms, condition number, machine epsilon, Rayleigh quotient)
1. The matvec is the unit of cost: HVP three ways
2. Power iteration
3. Deflation
4. Lanczos: the three-term recurrence
5. Loss of orthogonality, and how to fix it
6. Hessian top-k in practice

Solutions live in `solutions/_01_krylov.py`.  Inline tests live in `src/tests.py`.
"""))
    cells.append(code("""
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import torch
import matplotlib.pyplot as plt
import numpy as np

from src import tests
from src.plotting import apply_style, semilog_convergence, eigenvalue_compare
from src.tiny_models import toy_mlp, tiny_mlp, count_params

apply_style()
torch.manual_seed(0)
print('environment ready')
"""))


def _section_0_nla_prelim(cells):
    cells.append(md(r"""
## 0. NLA you'll need

This section establishes the four concepts the rest of the notebook leans on.
We state, don't derive — proofs in any numerical-linear-algebra textbook.

### 0.1 Operator vs Frobenius norm

For a matrix $A \in \mathbb{R}^{m \times n}$:

- **Operator norm** $\|A\|_2 = \sigma_{\max}(A)$.  Controls the worst-case
  matvec amplification: $\|Av\|_2 \le \|A\|_2 \|v\|_2$.
- **Frobenius norm** $\|A\|_F = \sqrt{\sum_{ij} A_{ij}^2} = \sqrt{\sum_i \sigma_i^2}$.
  Controls the *variance* of stochastic trace estimators.

Always $\|A\|_2 \le \|A\|_F \le \sqrt{\text{rank}(A)} \cdot \|A\|_2$.
"""))
    cells.append(code("""
# Exercise 0.1 (🔴⚪⚪⚪⚪, 3 min)
# Compute both norms of a 20x20 random matrix, verify the inequality.
torch.manual_seed(0)
A = torch.randn(20, 20)

op_norm = None  # YOUR CODE HERE: compute ||A||_2
fro_norm = None  # YOUR CODE HERE: compute ||A||_F

assert op_norm is not None and fro_norm is not None, 'fill in op_norm and fro_norm'
assert op_norm <= fro_norm <= (20 ** 0.5) * op_norm
print(f'||A||_2 = {op_norm:.3f}, ||A||_F = {fro_norm:.3f}')
"""))

    cells.append(md(r"""
### 0.2 Condition number

For a square invertible $A$: $\kappa(A) = \sigma_{\max}/\sigma_{\min}$.
For symmetric positive-definite $A$ this is $\lambda_{\max}/\lambda_{\min}$.

**Why it matters:** in `fp32`, solving $A x = b$ loses about $\log_{10}\kappa$
digits.  Iterative methods converge at a rate that depends on the eigenvalue
*gap*, which is bounded by $\kappa$.

Below: we make $\kappa$ progressively worse and watch the residual blow up.
"""))
    cells.append(code(r"""
# Exercise 0.2 (🔴🔴⚪⚪⚪, 8 min)
# Build a symmetric PSD A with engineered eigenvalues {1, 1, ..., 1, 1/kappa},
# solve Ax = b in fp32, plot relative residual vs kappa on log-log axes.

def make_conditioned_spd(n: int, kappa: float, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(n, n, generator=g))[0]
    eigs = torch.ones(n)
    eigs[-1] = 1.0 / kappa
    return Q @ torch.diag(eigs) @ Q.T

n = 30
kappas = torch.logspace(0, 8, 9)
residuals = []
for kappa in kappas:
    A = make_conditioned_spd(n, kappa.item()).float()
    x_true = torch.randn(n)
    b = A @ x_true
    x_hat = None  # YOUR CODE HERE: solve A x_hat = b in fp32
    assert x_hat is not None, 'solve A x_hat = b using torch.linalg.solve'
    residuals.append(((A @ x_hat - b).norm() / b.norm()).item())

plt.figure()
plt.loglog(kappas.numpy(), residuals, 'o-')
plt.xlabel(r'$\kappa(A)$'); plt.ylabel('relative residual')
plt.title('fp32 solve precision vs condition number')
plt.show()
"""))

    cells.append(md(r"""
### 0.3 Machine epsilon

`fp32` has $\epsilon_{\text{mach}} \approx 1.19 \times 10^{-7}$.
That's the smallest $x$ such that $1 + x$ is representable as
distinct from $1$.  Iterative methods compound rounding error,
and Lanczos (Section 5) is the textbook example of compounded
roundoff destroying a beautiful algorithm.
"""))
    cells.append(code("""
# Exercise 0.3 (🔴⚪⚪⚪⚪, 3 min)
# Find epsilon_machine empirically for fp32 and fp64.

def find_eps(dtype):
    one = torch.tensor(1.0, dtype=dtype)
    eps = torch.tensor(1.0, dtype=dtype)
    while one + eps / 2 > one:
        eps = eps / 2
    return eps.item()

eps32 = None  # YOUR CODE HERE
eps64 = None  # YOUR CODE HERE
assert eps32 is not None and eps64 is not None
print(f'eps(fp32) ≈ {eps32:.3e};  eps(fp64) ≈ {eps64:.3e}')
"""))

    cells.append(md(r"""
### 0.4 Rayleigh quotient & symmetric eigenvalues

For symmetric $A$ and unit $v$:

$$
R(v) = v^\top A v \in [\lambda_{\min}(A), \, \lambda_{\max}(A)].
$$

- $R(v_{\max}) = \lambda_{\max}$ when $v_{\max}$ is the top eigenvector.
- Power iteration is *ascent on $R$*: each step pushes $v$ further in the
  direction of largest $R$.

**Eigenvalues vs singular values.**  For symmetric $A$,
$\sigma_i(A) = |\lambda_i(A)|$ — singular values are absolute values of
eigenvalues.  For the Hessian, this matters: the **largest** eigenvalue and
the **largest-magnitude** eigenvalue can differ if the Hessian is indefinite
(common during training).  Power iteration converges to the largest-magnitude
one.
"""))
    cells.append(code("""
# Exercise 0.4 (🔴⚪⚪⚪⚪, 2 min)
# Verify: R(v) is bounded by [lambda_min, lambda_max] for random unit v.

torch.manual_seed(0)
A = torch.randn(15, 15); A = A + A.T
eigs = torch.linalg.eigvalsh(A)
lam_min, lam_max = eigs.min().item(), eigs.max().item()

for _ in range(20):
    v = torch.randn(15); v = v / v.norm()
    R = (v @ A @ v).item()
    assert lam_min - 1e-6 <= R <= lam_max + 1e-6, f'R={R} outside [{lam_min},{lam_max}]'
print(f'all 20 Rayleigh quotients in [{lam_min:.3f}, {lam_max:.3f}] ✓')
"""))


def _section_1_hvp(cells):
    pass


def _section_2_power(cells):
    pass


def _section_3_deflation(cells):
    pass


def _section_4_lanczos(cells):
    pass


def _section_5_orthogonality(cells):
    pass


def _section_6_hessian_topk(cells):
    pass


if __name__ == "__main__":
    nb = build()
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH} ({len(nb['cells'])} cells)")
