"""Build script for Notebook 3 (Hutchinson + SLQ + perturbation theory).

Run: `uv run python notebooks/build_03.py`
Output: notebooks/03_estimation.ipynb
"""
from __future__ import annotations

import nbformat as nbf

NB_PATH = "notebooks/03_estimation.ipynb"


def md(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_markdown_cell(s.strip("\n"))


def code(s: str) -> nbf.NotebookNode:
    return nbf.v4.new_code_cell(s.strip("\n"))


def build() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    cells: list[nbf.NotebookNode] = []
    _section_preamble(cells)
    _section_1_hutchinson(cells)
    _section_2_hessian_trace(cells)
    _section_3_slq(cells)
    _section_4_dos_in_practice(cells)
    _section_5_perturbation(cells)
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def _section_preamble(cells):
    cells.append(md("""
# Notebook 3 — Trace, density of states, and eigenvalue perturbation

Tools for *estimating* spectral quantities from matvecs alone.  Hutchinson
trace, stochastic Lanczos quadrature for the density of states, and the
perturbation theory (Weyl + Davis-Kahan) that tells you how much to trust
these estimates.

**Time:** ~115 min.  **Prerequisites:** Notebooks 1 & 2 (HVPs, Lanczos, rSVD).

**Sections:**
1. Hutchinson's trick
2. The Hessian's trace
3. From point estimates to the spectral density (SLQ)
4. DOS in practice — the bulk-plus-outliers picture
5. Eigenvalue perturbation: how much do you trust this?
"""))
    cells.append(code("""
import sys, os, math
sys.path.insert(0, os.path.abspath('..'))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from src.plotting import apply_style
from src.tiny_models import toy_mlp, tiny_mlp, count_params
from src.data import load_mnist_7x7
from solutions._01_krylov import hvp_double_backward
from solutions._02_randomized import entk_matvec

apply_style()
torch.manual_seed(0)
print('environment ready')
"""))


def _section_1_hutchinson(cells):
    cells.append(md(r"""
## 1. Hutchinson's trick

For any random vector $z$ with $\mathbb{E}[z z^\top] = I$,

$$
\mathbb{E}[z^\top A z] = \mathrm{tr}(A).
$$

So averaging $z^\top A z$ over many random probes estimates the trace, with
nothing more than matvecs.

### Variance

For symmetric $A$:
- **Gaussian** probes ($z \sim \mathcal{N}(0, I)$): $\mathrm{Var}(z^\top A z) = 2 \|A\|_F^2$.
- **Rademacher** probes ($z_i \in \{-1, +1\}$ uniform): $\mathrm{Var}(z^\top A z) = 2(\|A\|_F^2 - \|\mathrm{diag}(A)\|^2)$.

So Rademacher is **always at least as good** as Gaussian for symmetric $A$,
and strictly better when $A$ has diagonal mass.  The variance of the
$m$-sample mean is then at most $\frac{2 \|A\|_F^2}{m}$.
"""))
    cells.append(md(r"""
### Exercise 1.1: Implement Hutchinson (🔴🔴⚪⚪⚪, 8 min)

Return both the mean estimate **and** the list of per-probe values (we'll
need the per-probe variance in the next exercise).
"""))
    cells.append(code(r"""
def hutchinson_trace(matvec, n, m, probe_type='rademacher', seed=0):
    # YOUR CODE HERE
    raise NotImplementedError

# Sanity check on a 30x30 random symmetric matrix.
torch.manual_seed(0)
n = 30
A = torch.randn(n, n); A = A + A.T
true_trace = torch.diagonal(A).sum().item()

est, _ = hutchinson_trace(lambda v: A @ v, n=n, m=1000, probe_type='rademacher', seed=0)
print(f'true trace  = {true_trace:.3f}')
print(f'Hutchinson  = {est:.3f}')
"""))
    cells.append(md(r"""
### Exercise 1.2: Variance plot (🔴🔴⚪⚪⚪, 8 min)

For $m \in \{10, 30, 100, 300, 1000, 3000\}$, run Hutchinson many times
(say 50 trials) with Rademacher and Gaussian probes.  Plot the empirical
variance of the estimator vs $m$.  Overlay the theoretical bound
$2\|A\|_F^2 / m$.

**What you plot:** for each $m$, $\widehat{\mathrm{Var}}(\hat{t}_m) = \frac{1}{50}\sum_{r=1}^{50}(\hat{t}_m^{(r)} - \bar{t})^2$
where $\hat{t}_m^{(r)}$ is the $r$-th Hutchinson estimate with $m$ probes.
"""))
    cells.append(code(r"""
A_diag = A + 3 * torch.eye(n)  # add diagonal mass to expose Rademacher's edge
A_fro_sq = (A_diag ** 2).sum().item()

ms = [10, 30, 100, 300, 1000, 3000]
n_trials = 50
fig, ax = plt.subplots()
for ptype, color in [('rademacher', 'C0'), ('gaussian', 'C1')]:
    emp_var = []
    for m in ms:
        ts = [hutchinson_trace(lambda v: A_diag @ v, n=n, m=m, probe_type=ptype,
                                seed=s)[0] for s in range(n_trials)]
        emp_var.append(torch.tensor(ts).var().item())
    ax.loglog(ms, emp_var, 'o-', color=color, label=ptype)
# theoretical bound for Gaussian
ax.loglog(ms, [2 * A_fro_sq / m for m in ms], 'k--', alpha=0.6,
          label=r'$2\|A\|_F^2/m$ (Gaussian)')
ax.set_xlabel('m (probes)'); ax.set_ylabel('Var(estimate)')
ax.set_title('Hutchinson variance: Rademacher vs Gaussian')
ax.legend(); plt.show()
"""))


def _section_2_hessian_trace(cells):
    cells.append(md(r"""
## 2. The Hessian's trace

Apply Hutchinson to the Hessian of a tiny MLP.  $\mathrm{tr}(H)$ is the sum
of eigenvalues; for an overparameterized network most eigenvalues are near
zero, but a few outliers contribute the bulk of the trace.

This is the simplest stochastic spectral statistic — and the building block
for the density-of-states machinery in the next section.
"""))
    cells.append(code("""
# Set up tiny_mlp + small training data.
torch.manual_seed(0)
model = tiny_mlp(seed=0)
X_train, y_train = load_mnist_7x7(n=300, seed=0)
P = count_params(model)
print(f'tiny_mlp: P = {P} params')

def H_matvec(v):
    return hvp_double_backward(model, X_train, y_train, v)

# Hutchinson estimate vs explicit trace.
est, _ = hutchinson_trace(H_matvec, n=P, m=200, probe_type='rademacher', seed=0)
print(f'Hutchinson trace (m=200): {est:.4f}')

# Ground truth via materialized Hessian.
H_full = torch.stack([H_matvec(torch.eye(P)[i]) for i in range(P)])
true_trace = torch.diagonal((H_full + H_full.T) / 2).sum().item()
print(f'true trace:               {true_trace:.4f}')
"""))


def _section_3_slq(cells):
    cells.append(md(r"""
## 3. From point estimates to the spectral density

Hutchinson gives a *scalar* spectral statistic.  But often we want the
**density of states**: a smoothed histogram of eigenvalues.

**Stochastic Lanczos Quadrature (SLQ)** combines Hutchinson-style probing
with the Lanczos process.  For each probe $z$:

1. Run $s$-step Lanczos starting at $z$, getting a tridiagonal $T_s$.
2. Eigendecompose $T_s = U_T \Theta U_T^\top$.  The Ritz values $\theta_i$
   approximate a discrete spectral measure.
3. The first-component weights $\omega_i = (U_T)_{0,i}^2$ tell you how much
   each Ritz value should contribute (this is the Gauss quadrature weight).

Sum over probes, smooth with a Gaussian:

$$
\rho(\lambda) \approx \frac{n}{m \cdot \sigma\sqrt{2\pi}} \sum_{p=1}^{m} \sum_{i=1}^{s} \omega_i^{(p)} \exp\!\left(-\frac{(\lambda - \theta_i^{(p)})^2}{2\sigma^2}\right)
$$

Each probe costs $s$ matvecs.  Total: $m \cdot s$ matvecs.
"""))
    cells.append(md(r"""
### Exercise 3.1: Implement SLQ (🔴🔴🔴🔴⚪, 25 min)

You'll need Lanczos with **full reorthogonalization** (small `s`, so cost
is OK), since selective reorth's slight inaccuracy hurts the weights.

Use `from solutions._01_krylov import lanczos` if you don't want to redo it.
"""))
    cells.append(code(r"""
from solutions._01_krylov import lanczos

def slq_density(matvec, n, m_probes, s_lanczos, grid, sigma, seed=0):
    # YOUR CODE HERE
    raise NotImplementedError

# Sanity check: a diagonal matrix with known spectrum.
torch.manual_seed(0)
n = 40
A_diag = torch.diag(torch.linspace(-2, 4, n))
grid = torch.linspace(-3, 5, 400)
density = slq_density(lambda v: A_diag @ v, n=n, m_probes=30, s_lanczos=30,
                      grid=grid, sigma=0.15, seed=0)

# Compare to the true eigenvalues (Dirac comb) and a Gaussian-smoothed
# kernel-density estimate of them.
true_eigs = torch.linalg.eigvalsh(A_diag)
true_kde = torch.zeros_like(grid)
for e in true_eigs:
    true_kde += torch.exp(-((grid - e)**2)/(2*0.15**2)) / (0.15 * math.sqrt(2*math.pi))

fig, ax = plt.subplots()
ax.plot(grid.numpy(), density.numpy(), label='SLQ estimate')
ax.plot(grid.numpy(), true_kde.numpy(), '--', alpha=0.7, label='true (KDE on eigs)')
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$\rho(\lambda)$')
ax.set_title('SLQ vs true spectral density (40 eigvals, linspace(-2, 4))')
ax.legend(); plt.show()
"""))


def _section_4_dos_in_practice(cells):
    cells.append(md(r"""
## 4. Density of states in practice — the bulk + outliers picture

Now apply SLQ to the Hessian of `tiny_mlp` at init and after training.

**Money plot:** the spectral density on **log-y** axes.  At init, mostly
isotropic random Gaussian-like spectrum.  After training, a clear
**bulk near zero** (most eigenvalues collapse there) plus a few
**outliers** — the high-curvature directions Lanczos would have caught.

The plot shows $\rho(\lambda)$ as a function of $\lambda$, with the
density normalized so $\int \rho(\lambda) d\lambda = P$ (total number of
eigenvalues).
"""))
    cells.append(code(r"""
# === DOS at init ===
def H_init_matvec(v):
    return hvp_double_backward(model, X_train, y_train, v)

grid = torch.linspace(-2.0, 5.0, 500)
dos_init = slq_density(H_init_matvec, n=P, m_probes=10, s_lanczos=40,
                       grid=grid, sigma=0.05, seed=0)
print(f'integrated DOS at init: {torch.trapezoid(dos_init, grid).item():.1f}  (P = {P})')
"""))
    cells.append(code(r"""
# Train a bit.
opt = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(200):
    idx = torch.randint(0, len(X_train), (64,))
    opt.zero_grad()
    F.cross_entropy(model(X_train[idx]), y_train[idx]).backward()
    opt.step()
print(f'trained, final loss = {F.cross_entropy(model(X_train), y_train).item():.3f}')

def H_trained_matvec(v):
    return hvp_double_backward(model, X_train, y_train, v)

dos_trained = slq_density(H_trained_matvec, n=P, m_probes=10, s_lanczos=40,
                          grid=grid, sigma=0.05, seed=0)

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.semilogy(grid.numpy(), dos_init.clamp(min=1e-3).numpy(), label='at init')
ax.semilogy(grid.numpy(), dos_trained.clamp(min=1e-3).numpy(), label='after training')
ax.set_xlabel(r'$\lambda$'); ax.set_ylabel(r'$\rho(\lambda)$ (log)')
ax.set_title('Money plot: Hessian DOS — bulk near zero + outliers after training')
ax.legend(); plt.show()
"""))


def _section_5_perturbation(cells):
    cells.append(md(r"""
## 5. Eigenvalue perturbation: how much do you trust this?

Every estimate above is *implicitly* a perturbation argument: the matvec
we have is a noisy or sampled version of the matrix we care about.  Two
classical bounds tell us how spectral quantities respond.

### Weyl's inequality

For symmetric $A$ and perturbation $E$ (also symmetric),

$$
|\lambda_k(A + E) - \lambda_k(A)| \le \|E\|_2 \quad \forall k.
$$

Eigenvalues are **Lipschitz** in the matrix, with constant 1 in operator
norm.  Even with $\|E\|_2$ moderately large, the spectrum doesn't move much.

### Davis-Kahan $\sin \Theta$

Eigen*vectors* are a different story.  If $V_k$ and $\widetilde V_k$ are the
top-$k$ invariant subspaces of $A$ and $A + E$, the largest principal angle
$\Theta_k$ satisfies

$$
\sin \Theta_k \le \frac{\|E\|_2}{\mathrm{gap}_k},
$$

where $\mathrm{gap}_k = \lambda_k(A) - \lambda_{k+1}(A)$ — the spectral gap.
**Eigenvectors are unstable when gaps are small.**  This is the genuine
"PCA directions are noisy on small datasets" phenomenon, formalized.
"""))
    cells.append(md(r"""
### Exercise 5.1: Weyl in action (🔴🔴⚪⚪⚪, 8 min)

Pick a 50×50 symmetric $A$ and a perturbation $E$.  For
$\alpha \in [0, 1]$, compute $\lambda_k(A + \alpha E)$ for each $k$, and
plot $|\Delta \lambda_k|$ vs $\alpha \|E\|_2$.

**What you plot:** the $x$-axis is $\alpha \|E\|_2$, the $y$-axis is
$\max_k |\lambda_k(A + \alpha E) - \lambda_k(A)|$.  The Weyl bound is the
identity line $y = x$.  Verify points stay on or below.
"""))
    cells.append(code(r"""
torch.manual_seed(0)
A = torch.randn(50, 50); A = A + A.T
E = torch.randn(50, 50); E = E + E.T
E_norm = torch.linalg.matrix_norm(E, ord=2).item()
A_eigs = torch.linalg.eigvalsh(A)

alphas = torch.linspace(0, 1, 21)
max_dlams = []
for alpha in alphas:
    P_eigs = torch.linalg.eigvalsh(A + alpha * E)
    max_dlams.append((P_eigs - A_eigs).abs().max().item())

xs = (alphas * E_norm).tolist()
plt.figure()
plt.plot(xs, max_dlams, 'o-', label=r'observed $\max_k |\Delta\lambda_k|$')
plt.plot([0, max(xs)], [0, max(xs)], 'k--', alpha=0.6, label='Weyl bound $y=x$')
plt.xlabel(r'$\alpha \|E\|_2$'); plt.ylabel(r'$\max_k |\Delta \lambda_k|$')
plt.title("Weyl's inequality: eigenvalues are 1-Lipschitz in the matrix")
plt.legend(); plt.show()
"""))
    cells.append(md(r"""
### Exercise 5.2: Davis-Kahan and the gap (🔴🔴🔴⚪⚪, 12 min)

Construct $A$ with an adjustable spectral gap: top-3 eigenvalues
$\{10, 10 - g, 10 - 2g\}$ for some gap $g$; remaining 47 in $[0, 1]$.

For each $g \in \{0.1, 0.3, 1.0, 3.0\}$, perturb $A$ by a fixed small $E$
($\|E\|_2$ ≈ 0.1).  Measure the principal angle between $A$'s top-3
eigenspace and $(A + E)$'s top-3 eigenspace.  Plot principal angle vs
$g$.  Should diverge as $g \to 0$.

**What you plot:** $\sin \Theta_3$ on the $y$-axis, gap $g$ on the
$x$-axis (log-log).  The Davis-Kahan ceiling $\|E\|_2 / g$ is the slope-$-1$
upper bound.
"""))
    cells.append(code(r"""
def make_A_with_gap(gap, n=50, seed=0):
    g_rng = torch.Generator().manual_seed(seed)
    Q = torch.linalg.qr(torch.randn(n, n, generator=g_rng))[0]
    eigs = torch.cat([
        torch.tensor([10.0, 10.0 - gap, 10.0 - 2*gap]),
        torch.rand(n - 3, generator=g_rng),
    ])
    return Q @ torch.diag(eigs) @ Q.T

from solutions._03_estimation import principal_angle

torch.manual_seed(1)
E = torch.randn(50, 50); E = E + E.T
E = 0.05 * E / torch.linalg.matrix_norm(E, ord=2)
E_norm = torch.linalg.matrix_norm(E, ord=2).item()

gaps = [0.1, 0.3, 1.0, 3.0]
angles, ceilings = [], []
for gap in gaps:
    A = make_A_with_gap(gap)
    _, V_A = torch.linalg.eigh(A)
    _, V_pert = torch.linalg.eigh(A + E)
    top3_A = V_A[:, -3:]
    top3_pert = V_pert[:, -3:]
    angle = principal_angle(top3_A, top3_pert)
    angles.append(math.sin(angle))
    ceilings.append(E_norm / gap)

fig, ax = plt.subplots()
ax.loglog(gaps, angles, 'o-', label=r'observed $\sin\Theta_3$')
ax.loglog(gaps, ceilings, 'k--', alpha=0.6, label=r'Davis-Kahan: $\|E\|_2/\mathrm{gap}$')
ax.set_xlabel('spectral gap $g$'); ax.set_ylabel(r'$\sin \Theta_3$')
ax.set_title(r'Davis-Kahan: small gap $\Rightarrow$ eigenvector instability')
ax.legend(); plt.show()
"""))
    cells.append(md(r"""
### Exercise 5.3: Bootstrap on the eNTK — the ML punchline (🔴🔴🔴⚪⚪, 10 min)

Compute the top-5 eNTK eigenvalues and eigenvectors on a 200-sample MNIST
subset.  Then resample 20 times (with replacement), recompute top-5 each
time.

**Plot one:** bootstrap distribution of eigenvalues (boxplots, k=1..5).
Should be tight.

**Plot two:** for each pair of bootstrap samples, compute the principal
angle between their top-3 eigenspaces.  Histogram of these angles.
Should show significant rotation, even though eigenvalues are stable.

**Punchline:** eigenvalue summaries are bootstrap-robust; eigenvector
summaries are not, exactly where gaps are small.
"""))
    cells.append(code(r"""
from solutions._02_randomized import randomized_eigh

torch.manual_seed(0)
model_n = tiny_mlp(seed=0)
X_full, _ = load_mnist_7x7(n=200, seed=0)
N = X_full.shape[0]

n_boot = 20
boot_eigvals = []
boot_eigvecs = []
for b in range(n_boot):
    g = torch.Generator().manual_seed(b)
    idx = torch.randint(0, N, (N,), generator=g)  # bootstrap with replacement
    X_boot = X_full[idx]

    def Kv(v, X=X_boot):
        return entk_matvec(model_n, X, v)

    eigvals, eigvecs = randomized_eigh(Kv, n=N, k=5, oversample=10, n_power=1, seed=b)
    boot_eigvals.append(eigvals)
    boot_eigvecs.append(eigvecs[:, :3])  # keep top-3 for angle analysis

eigval_arr = torch.stack(boot_eigvals)  # (n_boot, 5)
print(f'top-5 eigenvalue means across {n_boot} bootstraps:')
print(eigval_arr.mean(dim=0).tolist())
print(f'top-5 eigenvalue stds:')
print(eigval_arr.std(dim=0).tolist())
"""))
    cells.append(code(r"""
# Plot one: boxplot of bootstrap eigenvalue distributions.
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
axes[0].boxplot([eigval_arr[:, k].tolist() for k in range(5)])
axes[0].set_xticklabels([f'λ_{k+1}' for k in range(5)])
axes[0].set_ylabel('eigenvalue'); axes[0].set_title('Eigenvalues: stable across bootstraps')

# Plot two: pairwise principal angles between bootstrap eigenspaces.
angles = []
for i in range(n_boot):
    for j in range(i+1, n_boot):
        try:
            a = principal_angle(boot_eigvecs[i], boot_eigvecs[j])
            angles.append(math.sin(a))
        except Exception:
            pass

axes[1].hist(angles, bins=20)
axes[1].set_xlabel(r'$\sin \Theta_3$ (pairwise)'); axes[1].set_ylabel('count')
axes[1].set_title('Eigenvectors: rotate freely (small spectral gaps)')
plt.tight_layout(); plt.show()
"""))
    cells.append(md(r"""
### Wrap-up

Three classes of spectral estimation are now in your toolbox:

| Tool                | Returns                | Cost            |
|---------------------|------------------------|-----------------|
| Hutchinson          | scalar trace           | m matvecs       |
| SLQ                 | smoothed DOS curve     | m·s matvecs     |
| Bootstrap + rSVD    | CI on top-k eigenpairs | (k+p) × bootstraps matvecs |

And two perturbation bounds to interpret them:

| Bound        | What it says                                  |
|--------------|-----------------------------------------------|
| Weyl         | $|\Delta\lambda_k| \le \|E\|_2$               |
| Davis-Kahan  | $\sin\Theta_k \le \|E\|_2 / \mathrm{gap}_k$   |

**Next:** Notebook 4 — capstone.  Everything from N1-N3 applied to a small
CNN, tracked across training.
"""))


if __name__ == "__main__":
    nb = build()
    with open(NB_PATH, "w") as f:
        nbf.write(nb, f)
    print(f"wrote {NB_PATH} ({len(nb['cells'])} cells)")
