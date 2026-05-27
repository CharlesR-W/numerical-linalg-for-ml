"""Build script for Notebook 4 (Capstone: spectroscopy of a network across training).

Run: `uv run python notebooks/build_04.py`
Output: notebooks/04_capstone.ipynb
"""
from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/04_capstone.ipynb"


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []
    _section_preamble(cells)
    _section_1_setup(cells)
    _section_2_top_k(cells)
    _section_3_trace_and_dos(cells)
    _section_4_entk_modes(cells)
    _section_5_synthesis(cells)
    _section_6_pointers(cells)
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def _section_preamble(cells):
    cells.append(md("""
# Notebook 4 — Capstone: spectroscopy of a network across training

Apply everything from Notebooks 1-3 to a tiny network at four checkpoints
(steps 0, 10, 100, 500 of SGD on 7×7 MNIST).  See how each spectral
fingerprint evolves.

**Time:** ~120 min.  **Prerequisites:** Notebooks 1, 2, 3.

**Sections:**
1. Setup — load checkpoints, sanity-check
2. Hessian top-k via Lanczos (across training)
3. Hessian trace + DOS via Hutchinson and SLQ
4. eNTK top eigenfunctions via rSVD
5. Synthesis: cost ledger, what's missing
6. Pointers: where to go next

Checkpoints are pre-computed by `scripts/train_capstone.py`.
"""))
    cells.append(code("""
import sys, os, math, time
sys.path.insert(0, os.path.abspath('..'))

from pathlib import Path
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

from src.plotting import apply_style
from src.tiny_models import tiny_mlp, count_params
from src.data import load_mnist_7x7
from solutions._01_krylov import hvp_double_backward, lanczos
from solutions._02_randomized import randomized_eigh, entk_matvec
from solutions._03_estimation import hutchinson_trace, slq_density

apply_style()
torch.manual_seed(0)
print('environment ready')
"""))


def _section_1_setup(cells):
    cells.append(md(r"""
## 1. Setup: load checkpoints

`scripts/train_capstone.py` trained a `tiny_mlp` for 500 SGD steps with
batch size 64, learning rate 0.1, on 500 samples of 7×7 MNIST.

Checkpoints saved at steps 0, 10, 100, 500 capture: initialization, early
training, mid training, late training.
"""))
    cells.append(code("""
# Resolve the checkpoint directory robustly: works whether the notebook is
# run from `notebooks/` (jupyter lab) or from the project root (nbclient).
for cand in [Path('../data/checkpoints'), Path('data/checkpoints')]:
    if cand.exists():
        CKPT_DIR = cand
        break
else:
    raise FileNotFoundError('Could not find data/checkpoints/; run scripts/train_capstone.py')

STEPS = [0, 10, 100, 500]
checkpoints = {}
for s in STEPS:
    ckpt = torch.load(CKPT_DIR / f'step_{s}.pt', weights_only=False)
    m = tiny_mlp(seed=0)
    m.load_state_dict(ckpt['state_dict'])
    checkpoints[s] = m
    print(f'step {s:3d}: loss = {ckpt[\"loss\"]:.4f}')

# Load the same training set the checkpoints were trained on.
X_train, y_train = load_mnist_7x7(n=500, seed=0)
P = count_params(checkpoints[0])
print(f'\\nmodel: P = {P} params')
"""))


def _section_2_top_k(cells):
    cells.append(md(r"""
## 2. Hessian top-k via Lanczos across training

For each checkpoint, run Lanczos with selective reorth for 40 steps and
extract the top-10 Ritz values.

Plot top-10 $|\lambda_k|$ as $k = 1, \ldots, 10$ at each training step.
"""))
    cells.append(code(r"""
def hessian_topk(model, X, y, k=10, lanczos_steps=40, seed=0):
    def H_matvec(v):
        return hvp_double_backward(model, X, y, v)
    P = count_params(model)
    ritz, _ = lanczos(H_matvec, dim=P, k=lanczos_steps, reorth='selective', seed=seed)
    return ritz.abs().sort(descending=True).values[:k]

topk_by_step = {}
for s in STEPS:
    topk_by_step[s] = hessian_topk(checkpoints[s], X_train, y_train, k=10, seed=0)
    print(f'step {s:3d}: top-3 |λ| = {topk_by_step[s][:3].tolist()}')
"""))
    cells.append(code(r"""
fig, ax = plt.subplots()
for s in STEPS:
    ax.plot(range(1, 11), topk_by_step[s].numpy(), 'o-', label=f'step {s}')
ax.set_xlabel('rank k'); ax.set_ylabel(r'$|\lambda_k|$')
ax.set_yscale('log'); ax.legend()
ax.set_title('Hessian top-10 eigenvalues across training')
plt.show()
"""))


def _section_3_trace_and_dos(cells):
    cells.append(md(r"""
## 3. Trace and DOS across training

**Plot 1:** trace estimate $\hat{\mathrm{tr}}(H)$ via Hutchinson (200 probes)
at each checkpoint, alongside $\|\nabla L\|^2$ for comparison.

**Plot 2 (the big one):** DOS via SLQ at each checkpoint, stacked.  Watch
the spectrum evolve from "diffuse Gaussian-like" at init to "bulk near
zero plus a few outliers" after training.
"""))
    cells.append(code(r"""
traces, grad_norms = {}, {}
for s in STEPS:
    model = checkpoints[s]
    def Hmv(v, m=model): return hvp_double_backward(m, X_train, y_train, v)
    est, _ = hutchinson_trace(Hmv, n=P, m=200, probe_type='rademacher', seed=0)
    traces[s] = est

    # ||grad L||^2
    for p in model.parameters(): p.grad = None
    F.cross_entropy(model(X_train), y_train).backward()
    g_flat = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
    grad_norms[s] = (g_flat ** 2).sum().item()

fig, ax = plt.subplots()
ax.plot(STEPS, [traces[s] for s in STEPS], 'o-', label=r'$\mathrm{tr}(H)$')
ax.plot(STEPS, [grad_norms[s] for s in STEPS], 's-', label=r'$\|\nabla L\|^2$')
ax.set_xlabel('SGD step'); ax.set_xscale('symlog')
ax.set_yscale('log'); ax.legend(); ax.set_title('Trace and gradient norm across training')
plt.show()
"""))
    cells.append(code(r"""
# DOS via SLQ at each checkpoint.
grid = torch.linspace(-2.0, 5.0, 400)
dos = {}
for s in STEPS:
    model = checkpoints[s]
    def Hmv(v, m=model): return hvp_double_backward(m, X_train, y_train, v)
    dos[s] = slq_density(Hmv, n=P, m_probes=8, s_lanczos=30,
                         grid=grid, sigma=0.06, seed=0)

fig, axes = plt.subplots(len(STEPS), 1, figsize=(8, 8), sharex=True)
for ax, s in zip(axes, STEPS):
    ax.semilogy(grid.numpy(), dos[s].clamp(min=1e-3).numpy())
    ax.set_ylabel(r'$\rho(\lambda)$')
    ax.set_title(f'step {s}')
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel(r'$\lambda$')
plt.suptitle('Hessian DOS evolution: bulk consolidates, outliers emerge', y=1.0)
plt.tight_layout(); plt.show()
"""))


def _section_4_entk_modes(cells):
    cells.append(md(r"""
## 4. eNTK top eigenfunctions across training

For each checkpoint, compute the top-6 eNTK eigenfunctions on a 200-sample
MNIST subset via rSVD (matrix-free `entk_matvec` + `randomized_eigh`).

Visualize each eigenfunction as a 7×7 image (weighted average of training
samples by eigenvector entries).  Are they becoming more "feature-like"
through training?
"""))
    cells.append(code(r"""
X_eval, _ = load_mnist_7x7(n=200, seed=42)
N = X_eval.shape[0]

eigvecs_by_step = {}
eigvals_by_step = {}
for s in STEPS:
    model = checkpoints[s]
    def Kv(v, m=model): return entk_matvec(m, X_eval, v)
    eigvals, eigvecs = randomized_eigh(Kv, n=N, k=6, oversample=10, n_power=1, seed=0)
    eigvecs_by_step[s] = eigvecs
    eigvals_by_step[s] = eigvals
    print(f'step {s:3d}: top-3 eNTK eigvals = {eigvals[:3].tolist()}')
"""))
    cells.append(code(r"""
fig, axes = plt.subplots(len(STEPS), 6, figsize=(12, 2*len(STEPS)))
for row, s in zip(axes, STEPS):
    for col, k in enumerate(range(6)):
        u = eigvecs_by_step[s][:, k]
        img = (u[:, None] * X_eval).sum(dim=0).reshape(7, 7)
        row[col].imshow(img, cmap='RdBu_r')
        row[col].set_xticks([]); row[col].set_yticks([])
        if col == 0:
            row[col].set_ylabel(f'step {s}', fontsize=9)
        if s == STEPS[0]:
            row[col].set_title(f'eigfn {k+1}', fontsize=9)
plt.suptitle('eNTK top-6 eigenfunctions across training')
plt.tight_layout(); plt.show()
"""))


def _section_5_synthesis(cells):
    cells.append(md(r"""
## 5. Synthesis

### Exercise 5.1: Cost ledger (🔴🔴⚪⚪⚪, 12 min)

Fill in the table below with concrete numbers for the experiments above.

| Computation                          | Matvecs | Wall clock | If P = 10^7? |
|--------------------------------------|---------|------------|--------------|
| Hessian top-10 via Lanczos (40 step) |         |            |              |
| Trace via Hutchinson (200 probes)    |         |            |              |
| DOS via SLQ (8 probes × 30 Lanczos)  |         |            |              |
| eNTK top-6 via rSVD (16+1·16)        |         |            |              |

Where would the budget bind if this were a real ImageNet model?
"""))
    cells.append(code(r"""
# YOUR CODE HERE: time each spectral computation and fill in the ledger.
# Helper: time a function `n_calls` times and return avg ms.
def time_calls(fn, n_calls=3):
    fn()  # warmup
    t0 = time.perf_counter()
    for _ in range(n_calls):
        fn()
    return (time.perf_counter() - t0) / n_calls * 1000

model = checkpoints[100]
def Hmv(v): return hvp_double_backward(model, X_train, y_train, v)

t_lanczos = time_calls(lambda: lanczos(Hmv, dim=P, k=40, reorth='selective', seed=0), n_calls=3)
t_hutch = time_calls(lambda: hutchinson_trace(Hmv, n=P, m=50, probe_type='rademacher', seed=0)[0], n_calls=3)

print(f'Lanczos 40-step:   {t_lanczos:.0f} ms')
print(f'Hutchinson 50 prb: {t_hutch:.0f} ms')
"""))
    cells.append(md(r"""
### Exercise 5.2: What's *missing* from these spectra (🔴🔴⚪⚪⚪, 8 min)

Name three things you can *not* read off the spectral summaries above:

1. **Eigenvector localization** — does the top Hessian eigenvector
   concentrate on a few neurons, or spread across the whole network?
   Top-k Lanczos gives you the vectors, but the *spatial structure of those
   vectors* isn't a single scalar.
2. **Anisotropy across data points** — the Hessian-on-the-full-dataset
   averages over examples.  Per-example Hessians can differ wildly; the
   averaged spectrum hides this.
3. **Off-diagonal coupling between layers** — block-structure information
   (e.g., "the top eigenvalue is concentrated in layer 3") doesn't appear
   in the global density of states.
"""))


def _section_6_pointers(cells):
    cells.append(md(r"""
## 6. Pointers: where to go from here

### Influence functions

To know how much a training example affects a prediction, you need
$\nabla_\theta f(x_{\text{test}})^\top H^{-1} \nabla_\theta L(x_{\text{train}})$.
That **inverse Hessian-vector product** is solved with **conjugate
gradient** (CG) — iterative linear solve.  Same matrix-free philosophy,
different algorithm.  Followup tutorial.

### K-FAC and friends

The full Hessian inverse is too expensive; **block-diagonal approximations**
of the Fisher (which approximates the Hessian for cross-entropy at low loss)
work in practice.  K-FAC, EKFAC, Shampoo, Sophia.  All require a Kronecker
factoring of per-layer gradient covariances.

### The NTK / lazy regime

If a network's eNTK doesn't change much during training (the "lazy" regime),
the network is effectively a kernel method with kernel $K^{NTK}(x_i, x_j) =
\langle \nabla_\theta f(x_i), \nabla_\theta f(x_j) \rangle$ at *init*.  Then
training dynamics reduces to kernel regression.

Our Section 4 plot lets you check this empirically: if the top-6
eigenfunctions look the same at step 0 and step 500, the network is in the
NTK regime.  If they change a lot, it's not.

### Final pointer

Most of what's in this tutorial is from these surveys:
- Halko, Martinsson, Tropp (2011): "Finding structure with randomness."
  The randomized SVD bible.
- Ubaru, Chen, Saad (2017): "Fast estimation of $\mathrm{tr}(f(A))$ via SLQ."
  The classic stochastic Lanczos quadrature reference.
- Ghorbani, Krishnan, Xiao (2019): "An investigation into neural net
  optimization via the empirical Hessian density."  The bulk-plus-outliers
  picture.
- Novak, Sohl-Dickstein, Schoenholz (2022): "Fast finite width neural
  tangent kernel."  The matrix-free eNTK trick.
- Park, Sohl-Dickstein, Le, Smith (2019): "The effect of network width on
  the performance of large-batch training."  Outlier eigenvalues during
  training.

Happy spectroscopy.
"""))


if __name__ == "__main__":
    nb = build()
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH} ({len(nb['cells'])} cells)")
