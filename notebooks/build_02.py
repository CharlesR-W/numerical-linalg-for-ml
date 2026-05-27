"""Build script for Notebook 2 (Randomized methods + the empirical NTK).

Run: `uv run python notebooks/build_02.py`
Output: notebooks/02_randomized.ipynb
"""
from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/02_randomized.ipynb"


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []
    _section_preamble(cells)
    _section_0_stable_rank(cells)
    _section_1_why_randomize(cells)
    _section_2_rsvd_toy(cells)
    _section_3_entk_matvec(cells)
    _section_4_rsvd_on_entk(cells)
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def _section_preamble(cells):
    cells.append(md("""
# Notebook 2 — Randomized methods and the empirical NTK

ARENA-style hands-on tutorial on randomized linear algebra, with the
**empirical NTK** Gram matrix of a tiny network as the running example.

**Time:** ~100 min.  **Prerequisites:** Notebook 1 (HVPs, Lanczos).

**Sections:**
0. When randomization wins (stable rank)
1. Why randomize (motivation)
2. Randomized SVD on a synthetic matrix
3. The eNTK without materializing — Novak et al.'s VJP+JVP trick
4. rSVD on the eNTK: top-k eigenfunctions of a tiny network

Solutions live in `solutions/_02_randomized.py`.
"""))
    cells.append(code("""
import sys, os
sys.path.insert(0, os.path.abspath('..'))

import time
import torch
import matplotlib.pyplot as plt
import numpy as np

from src.plotting import apply_style
from src.tiny_models import toy_mlp, tiny_mlp, count_params

apply_style()
torch.manual_seed(0)
print('environment ready')
"""))


def _section_0_stable_rank(cells):
    cells.append(md(r"""
## 0. When randomization wins: stable rank

Randomized SVD is **fast** but **approximate**.  When does the approximation
hold up?  The answer is governed by a single quantity:

$$
r_s(A) = \frac{\|A\|_F^2}{\|A\|_2^2} = \frac{\sum_i \sigma_i^2}{\sigma_1^2}
$$

This is the **stable rank** or **effective rank**.  It satisfies
$1 \le r_s(A) \le \mathrm{rank}(A)$, and:

- $r_s(A) \approx 1$ when one singular value dominates → rSVD wins big.
- $r_s(A) \approx n$ when all singular values are equal → no free lunch.

The Halko-Martinsson-Tropp bound says rSVD recovers the top-$k$ singular
subspace with error $\lesssim \sqrt{r_s(A_{>k})}$ relative error, where
$A_{>k}$ is the tail past rank $k$.  So a fast-decaying spectrum (small
tail stable rank) makes rSVD shine.
"""))
    cells.append(md(r"""
### Exercise 0.1: Compute stable rank for three spectra (🔴⚪⚪⚪⚪, 5 min)

Build three diagonal matrices with different decay:
- **fast:** $\sigma_i = 2^{-i}$  (exponential)
- **slow:** $\sigma_i = 1/(1+i)$  (algebraic)
- **flat:** $\sigma_i = 1$ for all $i$

Compute and compare their stable ranks.
"""))
    cells.append(code(r"""
def stable_rank(A):
    # YOUR CODE HERE
    raise NotImplementedError

n = 50
specs = {
    'fast (2^-i)':    torch.diag(2.0 ** (-torch.arange(n).float())),
    'slow (1/(1+i))': torch.diag(1.0 / (1.0 + torch.arange(n).float())),
    'flat (all 1)':   torch.diag(torch.ones(n)),
}

for name, A in specs.items():
    r_s = stable_rank(A)
    print(f'{name:>20s}: r_s = {r_s:.2f}  (rank = {n})')

print('\nExpect: fast ≈ 1.33, slow ≈ 4, flat = 50')
"""))


def _section_1_why_randomize(cells):
    cells.append(md(r"""
## 1. Why randomize?

Lanczos (Notebook 1) gives top-k eigenvalues exactly in $O(k)$ matvecs.
So why bother with randomization?

**Two reasons:**

1. **Lanczos matvecs are sequential.**  You can't compute $A q_1$ before
   $A q_0$ tells you what $q_1$ is.  Randomized methods sketch with a
   **batch** of vectors that can be applied to $A$ in parallel — well-suited
   to GPUs.

2. **Reorthogonalization is bookkeeping-heavy.**  Selective reorth saves
   work but adds branching.  Randomized methods do a single QR at the end
   on a small matrix, no per-step bookkeeping.

The trade-off: randomized methods are *approximate*, and the approximation
degrades with slow spectral decay.

This section's two motivations get formalized as the Halko-Martinsson-Tropp
algorithm in Section 2.
"""))


def _section_2_rsvd_toy(cells):
    cells.append(md(r"""
## 2. Randomized SVD on a toy matrix

For a symmetric matrix $A \in \mathbb{R}^{n \times n}$, the
Halko-Martinsson-Tropp range-finder is:

1. **Sketch:** $Y = A \Omega$, where $\Omega \in \mathbb{R}^{n \times (k+p)}$
   has iid Gaussian columns.  $p$ is "oversampling".
2. **(Optional) power iterations:** $Y \leftarrow A^q Y$ to amplify the
   dominant subspace.
3. **Orthonormalize:** $Q = \mathrm{QR}(Y)$.  Columns of $Q$ approximate the
   top-$(k+p)$-dimensional invariant subspace.
4. **Project:** $B = Q^\top A Q \in \mathbb{R}^{(k+p) \times (k+p)}$.
5. **Solve the small problem:** eigendecompose $B$, lift back via $Q$.

Cost: $(k+p)(2q+1)$ matvecs + $O(n(k+p)^2)$ for the QR and small eigh.
"""))
    cells.append(md(r"""
### Exercise 2.1: Implement randomized_eigh (🔴🔴🔴⚪⚪, 20 min)
"""))
    cells.append(code("""
def randomized_eigh(matvec, n, k, oversample=10, n_power=0, seed=0):
    # YOUR CODE HERE
    raise NotImplementedError

# Sanity check on a fast-decay matrix.
torch.manual_seed(0)
n_test = 100
true_eigs = 2.0 ** (-torch.arange(n_test).float())
Q_test = torch.linalg.qr(torch.randn(n_test, n_test))[0]
A_test = Q_test @ torch.diag(true_eigs) @ Q_test.T

eigvals, eigvecs = randomized_eigh(lambda v: A_test @ v, n=n_test, k=5,
                                    oversample=10, seed=0)
true_top5 = true_eigs.abs().sort(descending=True).values[:5]
print(f'rSVD top-5:  {eigvals.abs().tolist()}')
print(f'true top-5:  {true_top5.tolist()}')
"""))
    cells.append(md(r"""
### Exercise 2.2: Sweep oversampling on three decay regimes (🔴🔴⚪⚪⚪, 12 min)

For each of fast / slow / flat decay, plot the top-$k$ reconstruction error
as a function of oversampling $p \in \{0, 5, 10, 20\}$.

You should see:
- **Fast decay:** error drops sharply at small $p$; saturates near machine
  epsilon by $p = 5$.
- **Slow decay:** error decreases steadily but slowly; oversampling alone
  isn't enough.
- **Flat decay:** rSVD basically fails — there's no "dominant subspace" to
  find.
"""))
    cells.append(code(r"""
n = 100
k = 5

def make_decay(kind):
    if kind == 'fast': eigs = 2.0 ** (-torch.arange(n).float())
    elif kind == 'slow': eigs = 1.0 / (1.0 + torch.arange(n).float())
    elif kind == 'flat': eigs = torch.ones(n)
    Q = torch.linalg.qr(torch.randn(n, n))[0]
    return Q @ torch.diag(eigs) @ Q.T, eigs.sort(descending=True).values[:k]

fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
for ax, kind in zip(axes, ['fast', 'slow', 'flat']):
    torch.manual_seed(7)
    A, true_top = make_decay(kind)
    for p in [0, 5, 10, 20]:
        errs = []
        for trial in range(5):
            ev, _ = randomized_eigh(lambda v: A @ v, n=n, k=k, oversample=p, seed=trial)
            errs.append((ev.abs() - true_top).abs().max().item())
        ax.semilogy([p]*5, errs, 'o', alpha=0.5)
        ax.semilogy([p], [sum(errs)/5], 'k_', markersize=18)
    ax.set_title(f'{kind} decay')
    ax.set_xlabel('oversampling p'); ax.set_ylabel('top-k max abs err')
plt.tight_layout(); plt.show()
"""))
    cells.append(md(r"""
### Exercise 2.3: Power iterations rescue slow decay (🔴🔴⚪⚪⚪, 10 min)

On slow-decay matrices, oversampling helps marginally.  **Power iterations**
inside the sketch — applying $A$ multiple times to the sketch before QR —
amplify the dominant subspace.

Try $q \in \{0, 1, 2, 4\}$ on a slow-decay matrix.  Plot error vs $q$.
"""))
    cells.append(code(r"""
torch.manual_seed(0)
A_slow, true_top_slow = make_decay('slow')

fig, ax = plt.subplots()
for q in [0, 1, 2, 4]:
    errs = []
    for trial in range(5):
        ev, _ = randomized_eigh(lambda v: A_slow @ v, n=n, k=k,
                                oversample=5, n_power=q, seed=trial)
        errs.append((ev.abs() - true_top_slow).abs().max().item())
    ax.semilogy([q]*5, errs, 'o', alpha=0.5)
    ax.semilogy([q], [sum(errs)/5], 'k_', markersize=18)
ax.set_xlabel('power iterations q'); ax.set_ylabel('top-k max abs err')
ax.set_title('Power iterations amplify dominant subspace (slow-decay matrix)')
plt.show()
"""))


def _section_3_entk_matvec(cells):
    cells.append(md(r"""
## 3. The eNTK without materializing

The **empirical neural tangent kernel** of a network at parameters $\theta$
is the $N \times N$ Gram matrix

$$
K_{ij} = \nabla_\theta f(x_i)^\top \nabla_\theta f(x_j)
$$

(taking $f(x) = \sum_c \text{model}(x)_c$ for the scalar case).  This is
$N^2$ entries, but each entry is a $P$-dimensional inner product, so the
naive cost to *materialize* $K$ is $O(N \cdot P)$ memory and $N$
forward+backward sweeps.

For a network with $P = 10^7$ and $N = 10^4$, that's already $10^{11}$
parameters of memory — impossible.  **But we never need $K$ explicitly.**

### The Novak et al. trick

$K = J J^\top$ where $J \in \mathbb{R}^{N \times P}$ stacks the per-sample
gradients.  So $K v = J (J^\top v)$.  We compute this without forming $J$:

- **Step 1 (one VJP):** $g = J^\top v = \sum_j v_j \nabla_\theta f(x_j) \in \mathbb{R}^P$.
  This is $\nabla_\theta \sum_j v_j f(x_j)$ — a single backward sweep through
  the weighted sum.
- **Step 2 (one JVP):** $K v = J g$.  Row $i$ is $\nabla_\theta f(x_i) \cdot g$.
  This is a single JVP through the network, taking $g$ as the parameter
  tangent.

Total cost: **one forward + one backward + one forward**.  Independent of
$N$ and $P$.  This is the matvec we'll use everywhere.
"""))
    cells.append(md(r"""
### Exercise 3.1: Implement entk_matvec (🔴🔴🔴🔴⚪, 25 min)

Use `torch.func.grad` (for the VJP step) and `torch.func.jvp` (for the JVP
step).  Hint: both operate on dict-valued functions of parameters.

The signature: `entk_matvec(model, X, v) -> (N,) tensor`.
"""))
    cells.append(code("""
from torch.func import functional_call, grad, jvp, jacrev

def entk_matvec(model, X, v):
    # YOUR CODE HERE
    raise NotImplementedError

# Verify against the explicit eNTK on a tiny case.
def entk_explicit(model, X):
    params = {n: p.detach() for n, p in model.named_parameters()}
    def f_scalar(pd):
        return functional_call(model, pd, (X,)).sum(dim=-1)
    J_dict = jacrev(f_scalar)(params)
    J = torch.cat([j.reshape(j.shape[0], -1) for j in J_dict.values()], dim=1)
    return J @ J.T

torch.manual_seed(0)
model = toy_mlp(seed=1)
X = torch.randn(12, 20)
v = torch.randn(12)

K = entk_explicit(model, X)
Kv_explicit = K @ v
Kv_matrixfree = entk_matvec(model, X, v)

print(f'max |Δ| = {(Kv_explicit - Kv_matrixfree).abs().max():.2e}')
assert torch.allclose(Kv_explicit, Kv_matrixfree, atol=1e-4)
print('matrix-free matches explicit ✓')
"""))
    cells.append(md(r"""
### Exercise 3.2: The money plot — time vs N (🔴🔴⚪⚪⚪, 10 min)

Sweep $N \in \{8, 32, 128, 512\}$ on `toy_mlp` (P ≈ 250).  Time:
1. Matrix-free `entk_matvec` (one VJP + one JVP regardless of N).
2. Explicit $K = J J^\top$ via `jacrev`, then $K v$ (N backward sweeps to
   build J).

You should see matrix-free stay roughly **flat** while explicit grows.
"""))
    cells.append(code("""
torch.manual_seed(0)
model_t = toy_mlp(seed=1)
P = count_params(model_t)
print(f'model: P = {P} params')

Ns = [8, 32, 128, 512]
t_matrixfree, t_explicit = [], []
for N in Ns:
    X = torch.randn(N, 20)
    v = torch.randn(N)

    # warmup
    entk_matvec(model_t, X, v)
    t0 = time.perf_counter()
    for _ in range(3):
        entk_matvec(model_t, X, v)
    t_matrixfree.append((time.perf_counter() - t0) / 3)

    entk_explicit(model_t, X)
    t0 = time.perf_counter()
    for _ in range(3):
        K = entk_explicit(model_t, X)
        Kv = K @ v
    t_explicit.append((time.perf_counter() - t0) / 3)
    print(f'N={N:4d}: matrix-free {t_matrixfree[-1]*1e3:7.1f} ms,  '
          f'explicit {t_explicit[-1]*1e3:7.1f} ms')

plt.figure(figsize=(7, 4.5))
plt.loglog(Ns, t_matrixfree, 'o-', label='matrix-free (1 VJP + 1 JVP)')
plt.loglog(Ns, t_explicit,    's-', label='explicit (build J, then K v)')
plt.xlabel('N (number of samples)'); plt.ylabel('matvec time (s)')
plt.title('Money plot: eNTK matvec cost vs N')
plt.legend(); plt.show()
"""))


def _section_4_rsvd_on_entk(cells):
    cells.append(md(r"""
## 4. rSVD on the eNTK: top-k eigenfunctions

With matrix-free `entk_matvec` and `randomized_eigh`, we can now do something
that would have been impossible with explicit storage: compute the top-k
eigenpairs of the eNTK on a real dataset.

We use `tiny_mlp` on 200 samples of 7×7 MNIST.  $K$ is a 200×200 matrix
that we never materialize.  Top eigenfunctions, visualized as images, tell
us what feature directions the kernel prioritizes.
"""))
    cells.append(code("""
from src.data import load_mnist_7x7

torch.manual_seed(0)
model_m = tiny_mlp(seed=0)
X_mnist, y_mnist = load_mnist_7x7(n=200, seed=0)

# Define K's matvec via closures.
def K_matvec(v):
    return entk_matvec(model_m, X_mnist, v)

# Top-6 eigenfunctions via rSVD.
eigvals, eigvecs = randomized_eigh(K_matvec, n=200, k=6, oversample=10, n_power=1, seed=0)
print(f'top-6 eNTK eigenvalues: {eigvals.tolist()}')
"""))
    cells.append(md(r"""
### Exercise 4.1: Visualize top eigenfunctions as 7×7 images (🔴🔴⚪⚪⚪, 8 min)

Each eigenvector $u_k \in \mathbb{R}^{200}$ assigns a weight to each of the
200 training samples.  Form the "eigenfunction image" by computing
$\sum_i u_k[i] \cdot X[i]$ — a weighted average input.  Plot these as a 2×3
grid of 7×7 images.
"""))
    cells.append(code(r"""
fig, axes = plt.subplots(2, 3, figsize=(9, 6))
for ax, k_idx in zip(axes.flat, range(6)):
    u_k = eigvecs[:, k_idx]
    # YOUR CODE HERE: form an "eigenfunction image" from u_k and X_mnist.
    eig_img = (u_k[:, None] * X_mnist).sum(dim=0).reshape(7, 7)
    ax.imshow(eig_img, cmap='RdBu_r')
    ax.set_title(f'eigenfn {k_idx+1}\nλ = {eigvals[k_idx]:.2g}')
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle('Top-6 eNTK eigenfunctions (tiny_mlp at init)')
plt.tight_layout(); plt.show()
"""))
    cells.append(md(r"""
### Wrap-up

You can now compute top-k eigenpairs of any Gram matrix you can write a
matvec for, with no materialization.

**Where we are on the O() table:**

| Cost                             | Notebook 2                                      |
|----------------------------------|-------------------------------------------------|
| eNTK matvec (matrix-free)        | O(forward + backward) — flat in N               |
| rSVD, top-k, oversample p        | (k+p) matvecs + O(n(k+p)²) for QR/eigh          |
| rSVD with q power iters          | (k+p)(2q+1) matvecs                             |

**Next:** Notebook 3 covers Hutchinson trace estimation, stochastic Lanczos
quadrature, and the eigenvalue perturbation theory that tells you how much
to trust any of these estimates.
"""))


if __name__ == "__main__":
    nb = build()
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH} ({len(nb['cells'])} cells)")
