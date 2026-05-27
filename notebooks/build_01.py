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
    cells.append(md(r"""
## 1. The matvec is the unit of cost

The **Hessian-vector product** (HVP) `H @ v` is the only way we'll touch the
Hessian of a neural network — never the matrix itself.  Three equivalent
implementations, all running in $O(\text{forward pass})$ time.

**Setup:**
- model: `toy_mlp(seed=1)`, ~250 params
- loss: cross-entropy on a small batch

We have three ways to compute $H v$:
1. **Finite differences on the gradient:** $\frac{\nabla \mathcal{L}(\theta + \epsilon v) - \nabla \mathcal{L}(\theta - \epsilon v)}{2\epsilon}$
2. **Double backward (the Pearlmutter trick):** form the scalar $g^\top v$ and differentiate again.
3. **JVP of grad:** push $v$ forward through `grad`, using `torch.func.jvp`.
"""))
    cells.append(code("""
# Setup
torch.manual_seed(0)
model = toy_mlp(seed=1)
X = torch.randn(8, 20)
y = torch.randint(0, 4, (8,))
P = count_params(model)
v = torch.randn(P)
print(f'model: {P} params')
"""))
    cells.append(md(r"""
### Exercise 1.1: HVP via double backward (🔴🔴⚪⚪⚪, 10 min)

Implement `hvp_double_backward(model, X, y, v)`.  Hint: build a single flat
parameter tensor with `requires_grad=True`, recompute the loss with
`torch.func.functional_call`, take the first gradient with
`create_graph=True`, then take a second gradient of $g \cdot v$.
"""))
    cells.append(code("""
import torch.nn.functional as F
from torch.func import functional_call

def flat_params(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])

def unflatten_into_dict(flat, ref_named):
    out, i = {}, 0
    for n, p in ref_named.items():
        out[n] = flat[i:i+p.numel()].view_as(p)
        i += p.numel()
    return out

def hvp_double_backward(model, X, y, v):
    # YOUR CODE HERE
    raise NotImplementedError

tests.test_hvp(hvp_double_backward)
"""))
    cells.append(md(r"""
### Exercise 1.2: HVP via JVP of grad (🔴🔴🔴⚪⚪, 12 min)

Same answer, different machinery.  Use `torch.func.grad` to build a function
that returns the flat gradient, then `torch.func.jvp` to push `v` forward
through it.

The result of `jvp(f, (x,), (v,))` is `(f(x), df_x(v))`.  Use the second.
"""))
    cells.append(code("""
from torch.func import grad, jvp

def hvp_jvp_of_grad(model, X, y, v):
    # YOUR CODE HERE
    raise NotImplementedError

tests.test_hvp(hvp_jvp_of_grad)
"""))
    cells.append(md(r"""
### Exercise 1.3: HVP via finite differences (🔴⚪⚪⚪⚪, 5 min)

Central differences on `grad`.  Slow, noisy, but the reference-of-references
that we'll use to sanity-check the others.  Use $\epsilon = 10^{-3}$.
"""))
    cells.append(code("""
def hvp_finite_difference(model, X, y, v, eps=1e-3):
    # YOUR CODE HERE
    raise NotImplementedError

# Verify all three agree.
v_test = torch.randn(P)
h_dbl = hvp_double_backward(model, X, y, v_test)
h_jvp = hvp_jvp_of_grad(model, X, y, v_test)
h_fd  = hvp_finite_difference(model, X, y, v_test)

print(f'||double - jvp||_inf = {(h_dbl - h_jvp).abs().max():.2e}')
print(f'||double - fd||_inf  = {(h_dbl - h_fd).abs().max():.2e}  (FD is noisier)')
assert torch.allclose(h_dbl, h_jvp, atol=1e-5)
assert torch.allclose(h_dbl, h_fd,  atol=5e-3)
print('all three agree ✓')
"""))
    cells.append(md(r"""
### Exercise 1.4: Timing (🔴⚪⚪⚪⚪, 5 min)

Time all three HVPs.  Expected ordering: `jvp_of_grad` ≈ `double_backward`
(both single forward+backward); `finite_difference` ≈ 2× as costly
(two gradient computations).

**Punchline:** an HVP is one forward + one backward, independent of $P$.
That's why we never materialize the Hessian.
"""))
    cells.append(code("""
import time

def time_hvp(fn, n_calls=20):
    fn(model, X, y, v)  # warmup
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn(model, X, y, v)
    return (time.perf_counter() - t0) / n_calls

t_dbl = time_hvp(hvp_double_backward)
t_jvp = time_hvp(hvp_jvp_of_grad)
t_fd  = time_hvp(hvp_finite_difference)
print(f'double backward: {t_dbl*1e3:.2f} ms')
print(f'jvp of grad:     {t_jvp*1e3:.2f} ms')
print(f'finite diff:     {t_fd*1e3:.2f} ms (≈ 2× the others)')
"""))


def _section_2_power(cells):
    cells.append(md(r"""
## 2. Power iteration

The simplest matrix-free eigenvalue method.  Given a matvec, iterate:

$$
v_{k+1} = \frac{A v_k}{\|A v_k\|}, \qquad \lambda_k = v_k^\top A v_k
$$

The Rayleigh quotient $\lambda_k$ converges to the **largest-magnitude**
eigenvalue at rate $|\lambda_2/\lambda_1|^k$.  Small ratio = fast convergence,
nearly-degenerate spectrum = pathologically slow.
"""))
    cells.append(md(r"""
### Exercise 2.1: Implement power iteration (🔴🔴⚪⚪⚪, 8 min)

Return the top eigenvalue, the top eigenvector, and a *history* of per-step
absolute change in eigenvalue estimate (for the convergence plot below).
"""))
    cells.append(code("""
def power_iteration(matvec, dim, num_iters=200, tol=1e-10, seed=0):
    # YOUR CODE HERE
    raise NotImplementedError

tests.test_power_iteration(lambda mv, dim, num_iters, seed:
                            power_iteration(mv, dim, num_iters, seed=seed)[:2])
"""))
    cells.append(md(r"""
### Exercise 2.2: Convergence rate vs spectral gap (🔴🔴⚪⚪⚪, 10 min)

Build three diagonal SPD matrices with engineered spectra:
- **wide gap:** eigs = [10, 1, 1, ..., 1].  Ratio = 0.1, fast.
- **narrow gap:** eigs = [10, 9.5, 1, ..., 1].  Ratio = 0.95, slow.
- **tied:** eigs = [10, 10, 1, ..., 1].  Ratio = 1.0, no convergence.

Plot the per-step Rayleigh-quotient change on semilog axes.  Verify the
slope matches $\log_{10}|\lambda_2/\lambda_1|$.
"""))
    cells.append(code(r"""
def make_diag_spd(eigs):
    return torch.diag(torch.tensor(eigs, dtype=torch.float32))

specs = {
    'wide gap (λ₂/λ₁ = 0.1)':   [10.0] + [1.0]*49,
    'narrow gap (λ₂/λ₁ = 0.95)': [10.0, 9.5] + [1.0]*48,
    'tied (λ₂/λ₁ = 1.0)':        [10.0, 10.0] + [1.0]*48,
}

fig, ax = plt.subplots()
for name, eigs in specs.items():
    A = make_diag_spd(eigs)
    matvec = lambda v, A=A: A @ v
    _, _, hist = power_iteration(matvec, dim=50, num_iters=120, seed=0)
    # Avoid log(0) on semilogy when exact convergence hits.
    hist_safe = [max(h, 1e-30) for h in hist]
    ax.semilogy(hist_safe, label=name)
ax.set_xlabel('iteration'); ax.set_ylabel(r'$|\lambda^{(k)} - \lambda^{(k-1)}|$')
ax.legend(); ax.set_title('Power iteration: convergence vs spectral gap')
plt.show()
"""))
    cells.append(md(r"""
### Exercise 2.3: Power iteration on the Hessian (🔴🔴⚪⚪⚪, 7 min)

Use the toy MLP from Section 1.  Define a matvec that calls
`hvp_double_backward` and run power iteration to find the top eigenvalue of
the Hessian.

For comparison, materialize the full Hessian (you *can* for 250 params) and
verify against `torch.linalg.eigvalsh`.
"""))
    cells.append(code("""
def hessian_matvec_factory(model, X, y):
    def matvec(v):
        return hvp_double_backward(model, X, y, v)
    return matvec

matvec_H = hessian_matvec_factory(model, X, y)
top_eig, top_vec, _ = power_iteration(matvec_H, dim=P, num_iters=200, seed=0)
print(f'power iteration top |λ| = {top_eig:.4f}')

# Ground truth: materialize the Hessian using HVP on each basis vector.
H_full = torch.stack([matvec_H(torch.eye(P)[i]) for i in range(P)])
true_eigs = torch.linalg.eigvalsh((H_full + H_full.T) / 2)
print(f'true top |λ| = {true_eigs.abs().max():.4f}')
assert abs(abs(top_eig) - true_eigs.abs().max().item()) < 1e-3
print('✓ matches')
"""))


def _section_3_deflation(cells):
    cells.append(md(r"""
## 3. Deflation: getting top-k from power iteration

Once you have $(\lambda_1, v_1)$, subtract the rank-1 piece:

$$
A' = A - \lambda_1 v_1 v_1^\top
$$

Now run power iteration on $A'$ to get $(\lambda_2, v_2)$.  Repeat.

In matrix-free land, you don't form $A'$ explicitly.  You define a *deflated
matvec*:

```python
def deflated_matvec(v):
    out = matvec(v)
    for lam, u in zip(found_vals, found_vecs):
        out = out - lam * (u @ v) * u
    return out
```
"""))
    cells.append(md(r"""
### Exercise 3.1: Deflated power iteration (🔴🔴⚪⚪⚪, 10 min)

Recover the top-3 eigenpairs.  Verify against `torch.linalg.eigvalsh` on the
materialized Hessian.

**Warning:** naive deflation is fragile.  After ~5 deflations, accumulated
orthogonality error in the found eigenvectors makes the deflated operator
drift.  This is what Lanczos (next section) fixes.
"""))
    cells.append(code(r"""
def power_iteration_deflated(matvec, dim, k, num_iters_per=300, seed=0):
    # YOUR CODE HERE
    raise NotImplementedError

eigvals, eigvecs = power_iteration_deflated(matvec_H, dim=P, k=3, num_iters_per=400, seed=0)
# Compare by magnitude — power iteration finds largest-|λ|, and the Hessian
# may be indefinite at this point in training.
top3_by_mag = true_eigs.abs().sort(descending=True).values[:3]
print(f'top-3 by power+deflation (signed):    {[f"{x:+.3f}" for x in eigvals]}')
print(f'top-3 by torch.linalg (|λ|, sorted):  {top3_by_mag.tolist()}')
assert torch.allclose(
    torch.tensor([abs(e) for e in eigvals]).sort(descending=True).values,
    top3_by_mag, atol=1e-2,
)
print('✓ magnitudes match')
"""))
    cells.append(md(r"""
### Exercise 3.2: Watch deflation degrade (🔴🔴⚪⚪⚪, 7 min)

Push deflation to $k = 10$.  Plot relative error of each successive
eigenvalue.  Observe degradation past $k \approx 5$.
"""))
    cells.append(code(r"""
k_max = 10
eigvals_k, _ = power_iteration_deflated(matvec_H, dim=P, k=k_max, num_iters_per=300, seed=0)
true_topk = true_eigs.abs().sort(descending=True).values[:k_max].tolist()

rel_err = [abs(abs(e) - t) / t for e, t in zip(eigvals_k, true_topk)]
plt.figure()
plt.semilogy(range(1, k_max+1), rel_err, 'o-')
plt.xlabel('eigenvalue index'); plt.ylabel('relative error')
plt.title('Naive deflation degrades past ~5 eigenvalues')
plt.show()
"""))


def _section_4_lanczos(cells):
    cells.append(md(r"""
## 4. Lanczos: the three-term recurrence

Lanczos builds an orthonormal basis for the **Krylov subspace**

$$
\mathcal{K}_k(A, b) = \operatorname{span}\{b, Ab, A^2 b, \ldots, A^{k-1}b\}
$$

via a three-term recurrence.  The matrix $A$ projected into this basis is
*tridiagonal*: $T_k = Q_k^\top A Q_k$, where $Q_k = [q_0, q_1, \ldots, q_{k-1}]$.

The eigenvalues of $T_k$ — called **Ritz values** — converge to the
**extremes** of $A$'s spectrum first.

**Why "extremes first"?**  $\mathcal{K}_k$ is the space of degree-$(k-1)$
polynomials in $A$ applied to $b$.  Polynomials approximate well-separated
points easily; in-bulk eigenvalues are harder to resolve.

**The recurrence:**

$$
\beta_{j+1} q_{j+1} = A q_j - \alpha_j q_j - \beta_j q_{j-1}
$$

with $\alpha_j = q_j^\top A q_j$ and $\beta_{j+1} = \|r_{j+1}\|$.
"""))
    cells.append(md(r"""
### Exercise 4.1: Implement Lanczos without reorthogonalization (🔴🔴🔴⚪⚪, 20 min)

Return `(ritz_vals, Q)` where `Q` is the (dim × k) basis.  Use
`torch.linalg.eigvalsh(T)` to get the Ritz values from the tridiagonal.
"""))
    cells.append(code("""
def lanczos_no_reorth(matvec, dim, k, seed=0):
    # YOUR CODE HERE
    raise NotImplementedError

# Verify on a small dense matrix where we can compute ground truth.
torch.manual_seed(42)
A_dense = torch.randn(60, 60); A_dense = A_dense + A_dense.T
matvec_dense = lambda v: A_dense @ v
true_eigs_dense = torch.linalg.eigvalsh(A_dense)

ritz, Q = lanczos_no_reorth(matvec_dense, dim=60, k=20)
top5 = ritz.sort(descending=True).values[:5]
true5 = true_eigs_dense.sort(descending=True).values[:5]
print(f'top-5 Ritz:  {top5.tolist()}')
print(f'top-5 true:  {true5.tolist()}')
print(f'max |Δ|: {(top5 - true5).abs().max():.2e}')
"""))
    cells.append(md(r"""
### Exercise 4.2: Watch Ritz values converge (🔴🔴⚪⚪⚪, 8 min)

Run Lanczos for $k \in \{5, 10, 20, 40\}$ on the same matrix.  Plot true
eigenvalues vs Ritz values at each $k$.  Watch the extremes converge first.
"""))
    cells.append(code("""
fig, axes = plt.subplots(1, 4, figsize=(15, 4), sharey=True)
for ax, k in zip(axes, [5, 10, 20, 40]):
    ritz, _ = lanczos_no_reorth(matvec_dense, dim=60, k=k)
    eigenvalue_compare(true_eigs_dense, ritz, ax=ax)
    ax.set_title(f'k = {k}')
plt.suptitle('Lanczos Ritz values converge to extremes first')
plt.tight_layout(); plt.show()
"""))


def _section_5_orthogonality(cells):
    cells.append(md(r"""
## 5. Loss of orthogonality — and how to fix it

The Lanczos recurrence is mathematically beautiful but **numerically a trap**.
In exact arithmetic, $Q^\top Q = I$.  In `fp32`, accumulated rounding error
makes the columns of $Q$ drift out of orthogonality.  Once they do, Ritz
values start appearing as **ghost copies** — the algorithm "rediscovers" the
top eigenvalue multiple times.

The **money plot** of this section: $\|Q^\top Q - I\|_\infty$ blowing up
around step 20-30 in `fp32` for our 40×40 test matrix.
"""))
    cells.append(md(r"""
### Exercise 5.1: Watch orthogonality decay (🔴🔴⚪⚪⚪, 8 min)

Modify your `lanczos_no_reorth` to **also return $\|Q[:, :j+1]^\top Q[:, :j+1] - I\|_\infty$
at every step**.  Run for 40 steps on a 40×40 random symmetric matrix in
`fp32`.  Plot the orthogonality error vs step.
"""))
    cells.append(code(r"""
def lanczos_track_orth(matvec, dim, k, reorth='none', seed=0):
    '''As above but also returns orthogonality-error history.'''
    # YOUR CODE HERE: copy your lanczos_no_reorth and add per-step orth tracking.
    raise NotImplementedError

A40 = (torch.randn(40, 40, generator=torch.Generator().manual_seed(7))).float()
A40 = A40 + A40.T

_, _, orth_none = lanczos_track_orth(lambda v: A40 @ v, dim=40, k=40, reorth='none')

plt.figure()
plt.semilogy(orth_none, label='no reorth (fp32)')
plt.xlabel('step'); plt.ylabel(r'$\| Q^\top Q - I \|_\infty$')
plt.title('Orthogonality loss in classical Lanczos')
plt.legend(); plt.show()
"""))
    cells.append(md(r"""
### Exercise 5.2: Full reorthogonalization (🔴🔴⚪⚪⚪, 7 min)

At each step, subtract the projection of $r_{j+1}$ onto every previous
$q_i$ — and do it **twice** ("twice is enough", Kahan).  Add this as a
`reorth='full'` branch.

Cost: $O(k^2)$ extra work, which dominates for large $k$ but is fine for the
~30 steps we typically run.
"""))
    cells.append(code(r"""
_, _, orth_full = lanczos_track_orth(lambda v: A40 @ v, dim=40, k=40, reorth='full')

plt.figure()
plt.semilogy(orth_none, label='no reorth')
plt.semilogy(orth_full, label='full reorth (twice)')
plt.xlabel('step'); plt.ylabel(r'$\| Q^\top Q - I \|_\infty$')
plt.legend(); plt.title('Full reorth keeps orthogonality at fp32 epsilon')
plt.show()
"""))
    cells.append(md(r"""
### Exercise 5.3: Selective reorthogonalization (🔴🔴🔴⚪⚪, 12 min)

Full reorth wastes work most steps.  **Selective reorth** (Paige) only fires
when the residual norm drops far below what it "should" be.  Heuristic:
if $\|r_{j+1}\| < 0.717 \|A q_j\|$, reorthogonalize once.

Add `reorth='selective'`.  Plot all three on the same axes.

**Money plot** — this is the figure you'd put in a paper.
"""))
    cells.append(code(r"""
_, _, orth_sel = lanczos_track_orth(lambda v: A40 @ v, dim=40, k=40, reorth='selective')

plt.figure(figsize=(8, 4.5))
plt.semilogy(orth_none, label='none')
plt.semilogy(orth_full, label='full (twice-is-enough)')
plt.semilogy(orth_sel,  label='selective (Paige)')
plt.axhline(1e-7, color='k', linestyle=':', alpha=0.6, label='fp32 eps')
plt.xlabel('Lanczos step'); plt.ylabel(r'$\| Q^\top Q - I \|_\infty$')
plt.title('Money plot: orthogonality loss across reorth strategies')
plt.legend(); plt.show()
"""))


def _section_6_hessian_topk(cells):
    cells.append(md(r"""
## 6. Hessian top-k in practice

Bringing it together: Lanczos with selective reorth on the Hessian of the
**tiny MLP** trained for a few steps on 7×7 MNIST.

We compute the top-10 eigenvalues two ways:
1. Matrix-free Lanczos via our HVP.
2. Materialize the full ~2k × ~2k Hessian, eigendecompose explicitly.

Compare — they should match.  And then look at the same Hessian *before*
training vs *after*: outliers appear, the bulk consolidates.
"""))
    cells.append(code("""
from src.data import load_mnist_7x7

# Set up tiny_mlp + a small training run (just enough to see Hessian change).
torch.manual_seed(0)
model_mnist = tiny_mlp(seed=0)
X_train, y_train = load_mnist_7x7(n=500, seed=0)
P_mnist = count_params(model_mnist); print(f'tiny_mlp: {P_mnist} params')

# === Lanczos top-10 at init ===
matvec_init = lambda v: hvp_double_backward(model_mnist, X_train, y_train, v)
# YOUR CODE HERE: call lanczos_track_orth with reorth='selective', k=30,
# unpack the first element (Ritz values) into ritz_init.
ritz_init = None
assert ritz_init is not None and len(ritz_init) >= 10
print(f'top-10 |λ| at init (Lanczos): {sorted(ritz_init.abs().tolist(), reverse=True)[:10]}')
"""))
    cells.append(code("""
# Ground-truth check: materialize the Hessian (P x P), eigendecompose.
H_full_init = torch.stack([hvp_double_backward(model_mnist, X_train, y_train, torch.eye(P_mnist)[i])
                          for i in range(P_mnist)])
H_full_init = (H_full_init + H_full_init.T) / 2
true_eigs_init = torch.linalg.eigvalsh(H_full_init)
true_top10 = true_eigs_init.abs().sort(descending=True).values[:10]
print(f'top-10 |λ| at init (explicit): {true_top10.tolist()}')
"""))
    cells.append(md(r"""
### Exercise 6.1: Train a bit, recompute (🔴🔴⚪⚪⚪, 10 min)

Run 200 SGD steps with lr=0.1, batch size 64.  Recompute top-10 eigenvalues.
Plot init-vs-trained on the same axes.
"""))
    cells.append(code(r"""
opt = torch.optim.SGD(model_mnist.parameters(), lr=0.1)
for step in range(200):
    idx = torch.randint(0, len(X_train), (64,))
    opt.zero_grad()
    loss = F.cross_entropy(model_mnist(X_train[idx]), y_train[idx])
    loss.backward()
    opt.step()
print(f'final loss: {loss.item():.4f}')

matvec_trained = lambda v: hvp_double_backward(model_mnist, X_train, y_train, v)
ritz_trained = None  # YOUR CODE HERE: same as ritz_init but for the trained model.
assert ritz_trained is not None

top10_init = sorted(ritz_init.abs().tolist(), reverse=True)[:10]
top10_trained = sorted(ritz_trained.abs().tolist(), reverse=True)[:10]

fig, ax = plt.subplots()
ax.plot(range(1, 11), top10_init,    'o-', label='at init')
ax.plot(range(1, 11), top10_trained, 's-', label='after training')
ax.set_xlabel('rank'); ax.set_ylabel(r'$|\lambda_k|$')
ax.set_yscale('log'); ax.legend()
ax.set_title('Hessian top-10 eigenvalues: init vs trained')
plt.show()
"""))
    cells.append(md(r"""
### Wrap-up

You can now compute the top-k spectrum of any Hessian you can take an HVP of.

**Where we are on the O() table:**

| Cost                          | Notebook 1                              |
|-------------------------------|-----------------------------------------|
| matvec ≈ O(forward + backward)| HVP, three implementations              |
| power iter top-1              | k = O(log(1/ε)/log(λ₁/λ₂)) matvecs       |
| power iter + deflation top-k  | k×above; degrades past ~5 due to drift   |
| Lanczos top-k                 | O(k) matvecs + O(k²) for reorth          |

**Next:** Notebook 2 covers randomized methods and the empirical NTK.
"""))


if __name__ == "__main__":
    nb = build()
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH} ({len(nb['cells'])} cells)")
