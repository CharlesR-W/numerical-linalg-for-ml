# Phase 1 Implementation Plan: Scaffolding + Notebook 1 (Krylov)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stand up the `numerical-linalg-for-ml` tutorial repository and ship Notebook 1 (Krylov methods for the Hessian) as a working ARENA-style Jupyter exercise.

**Architecture:** `uv`-managed Python project under `~/Programming/Claude/tutorials/numerical-linalg-for-ml/`.  Pure PyTorch + `torch.func` for autodiff.  Reference implementations live in `solutions/01_krylov.py` (also the source of truth for later notebooks' ground-truth checks).  Notebooks are `.ipynb` files generated programmatically with `nbformat` so the build is reproducible and reviewable in git.  Each exercise stub has an inline `assert`-based test that calls into `solutions/`.

**Tech Stack:** Python 3.11, PyTorch 2.x with `torch.func`, NumPy, matplotlib, nbformat, pytest, ruff, uv.

**Scope:** Phase 1 only — scaffolding + Notebook 1.  Plans for Notebooks 2-4 (Randomized, Estimation+Perturbation, Capstone) will follow after Notebook 1 is reviewed and calibrated.

---

## File structure created in this phase

```
tutorials/numerical-linalg-for-ml/
├── pyproject.toml          # uv project, deps pinned
├── .gitignore              # ignore .ipynb_checkpoints, __pycache__, data/cache
├── README.md               # one-pager: how to run, O() table, money plots list
├── design.md               # already exists
├── src/
│   ├── __init__.py
│   ├── tiny_models.py      # toy_mlp(), tiny_mlp(), seeding helpers
│   ├── data.py             # MNIST-7x7 cached loader for tiny_mlp
│   ├── plotting.py         # mpl style, "money plot" helpers
│   └── tests.py            # tests.test_hvp(fn), tests.test_lanczos(fn), ...
├── solutions/
│   ├── __init__.py
│   └── 01_krylov.py        # reference HVP, power iter, deflate, Lanczos
├── notebooks/
│   ├── build_01.py         # script that writes 01_krylov.ipynb via nbformat
│   └── 01_krylov.ipynb     # generated; committed
├── tests/
│   ├── test_hvp.py         # unit tests for reference HVP
│   ├── test_power.py       # unit tests for power iter + deflation
│   └── test_lanczos.py     # unit tests for Lanczos + reorth
└── data/
    └── .gitkeep            # cache dir; actual data files .gitignored
```

**Why `notebooks/build_01.py` instead of editing `.ipynb` directly:** The `.ipynb` JSON is unreadable in diffs and hostile to edits.  Build script is the source of truth; `.ipynb` is a generated artifact that we commit so users can open it without installing anything.

---

## Phase A — Project scaffolding

### Task A1: Initialize uv project

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/pyproject.toml`
- Create: `tutorials/numerical-linalg-for-ml/.gitignore`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "numerical-linalg-for-ml"
version = "0.1.0"
description = "Hands-on numerical linear algebra tutorial for ML researchers"
requires-python = ">=3.11"
dependencies = [
    "torch>=2.2",
    "numpy>=1.26",
    "matplotlib>=3.8",
    "nbformat>=5.10",
    "jupyterlab>=4.0",
]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
]

[tool.ruff]
line-length = 88
target-version = "py311"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v"
```

- [ ] **Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.pytest_cache/
.ruff_cache/
.ipynb_checkpoints/
.venv/
data/*.pt
data/MNIST/
!data/.gitkeep
```

- [ ] **Step 3: Run `uv sync` to create the lock file**

Run: `cd ~/Programming/Claude/tutorials/numerical-linalg-for-ml && uv sync`
Expected: `.venv/` created, `uv.lock` created.

- [ ] **Step 4: Sanity-check the install**

Run: `uv run python -c "import torch; print(torch.__version__)"`
Expected: prints a 2.x version, no errors.

- [ ] **Step 5: Init git and commit**

```bash
cd ~/Programming/Claude/tutorials/numerical-linalg-for-ml
git init
git add pyproject.toml uv.lock .gitignore design.md plan-01-scaffolding-and-krylov.md
git commit -m "chore: initialize uv project for nla-for-ml tutorial"
```

---

### Task A2: Create directory skeleton

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/src/__init__.py` (empty)
- Create: `tutorials/numerical-linalg-for-ml/solutions/__init__.py` (empty)
- Create: `tutorials/numerical-linalg-for-ml/notebooks/.gitkeep`
- Create: `tutorials/numerical-linalg-for-ml/tests/.gitkeep`
- Create: `tutorials/numerical-linalg-for-ml/data/.gitkeep`

- [ ] **Step 1: Create all four `__init__.py` / `.gitkeep` files as empty**

- [ ] **Step 2: Commit**

```bash
git add src/__init__.py solutions/__init__.py notebooks/.gitkeep tests/.gitkeep data/.gitkeep
git commit -m "chore: create package directory skeleton"
```

---

## Phase B — Shared infrastructure

### Task B1: Implement `tiny_models.py`

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/src/tiny_models.py`
- Create: `tutorials/numerical-linalg-for-ml/tests/test_models.py`

- [ ] **Step 1: Write failing test**

`tests/test_models.py`:

```python
import torch
from src.tiny_models import toy_mlp, tiny_mlp, count_params


def test_toy_mlp_param_count():
    m = toy_mlp()
    assert 200 <= count_params(m) <= 350, "toy_mlp must stay ~250 params"


def test_tiny_mlp_param_count():
    m = tiny_mlp()
    assert 1500 <= count_params(m) <= 3000, "tiny_mlp ~2k params"


def test_toy_mlp_deterministic():
    a = toy_mlp(seed=0)
    b = toy_mlp(seed=0)
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb)


def test_toy_mlp_forward_shape():
    m = toy_mlp()
    x = torch.randn(3, 20)
    assert m(x).shape == (3, 4)


def test_tiny_mlp_forward_shape():
    m = tiny_mlp()
    x = torch.randn(3, 49)
    assert m(x).shape == (3, 10)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.tiny_models'`.

- [ ] **Step 3: Implement `tiny_models.py`**

```python
"""Tiny deterministic models used throughout the tutorial.

Three sizes:
- toy_mlp:  ~250 params.  Hessian materializable.  Ground-truth checks.
- tiny_mlp: ~2k params.  Hessian still materializable but tighter.  Workhorse.
- tiny_cnn: defined in a later phase for the capstone.
"""
import torch
import torch.nn as nn


def _seed_init(module: nn.Module, seed: int) -> None:
    """Seed PyTorch and reinitialize the module's parameters."""
    g = torch.Generator().manual_seed(seed)
    for p in module.parameters():
        if p.dim() >= 2:
            nn.init.kaiming_normal_(p, generator=g)
        else:
            nn.init.zeros_(p)


def toy_mlp(seed: int = 0) -> nn.Module:
    """20-dim input, width 8, depth 2, 4-class output.  ~250 params."""
    m = nn.Sequential(
        nn.Linear(20, 8),
        nn.Tanh(),
        nn.Linear(8, 8),
        nn.Tanh(),
        nn.Linear(8, 4),
    )
    _seed_init(m, seed)
    return m


def tiny_mlp(seed: int = 0) -> nn.Module:
    """49-dim input (7x7 MNIST), width 32, depth 2, 10-class.  ~2k params."""
    m = nn.Sequential(
        nn.Linear(49, 32),
        nn.Tanh(),
        nn.Linear(32, 32),
        nn.Tanh(),
        nn.Linear(32, 10),
    )
    _seed_init(m, seed)
    return m


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/tiny_models.py tests/test_models.py
git commit -m "feat(src): tiny_models with toy_mlp (~250p) and tiny_mlp (~2k)"
```

---

### Task B2: Implement `data.py` (MNIST 7×7 cached loader)

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/src/data.py`
- Create: `tutorials/numerical-linalg-for-ml/tests/test_data.py`

- [ ] **Step 1: Write failing test**

`tests/test_data.py`:

```python
import torch
from src.data import load_mnist_7x7


def test_load_mnist_7x7_shape():
    X, y = load_mnist_7x7(n=200, seed=0)
    assert X.shape == (200, 49)
    assert y.shape == (200,)
    assert X.dtype == torch.float32
    assert y.dtype == torch.long


def test_load_mnist_7x7_deterministic():
    X1, y1 = load_mnist_7x7(n=50, seed=42)
    X2, y2 = load_mnist_7x7(n=50, seed=42)
    assert torch.equal(X1, X2)
    assert torch.equal(y1, y2)


def test_load_mnist_7x7_normalized():
    X, _ = load_mnist_7x7(n=100, seed=0)
    assert -0.5 <= X.min() <= 0.0
    assert 0.5 <= X.max() <= 1.5
```

- [ ] **Step 2: Run test, verify it fails**

Run: `uv run pytest tests/test_data.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `data.py`**

```python
"""Tiny cached MNIST loader downsampled to 7x7 = 49 features.

Caches to data/mnist_7x7_n{N}.pt to avoid re-downloading on every notebook run.
"""
from pathlib import Path

import torch
import torch.nn.functional as F

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _download_full_mnist() -> tuple[torch.Tensor, torch.Tensor]:
    """One-time torchvision pull; cached as full_mnist.pt."""
    full_cache = DATA_DIR / "full_mnist.pt"
    if full_cache.exists():
        blob = torch.load(full_cache)
        return blob["X"], blob["y"]
    from torchvision import datasets, transforms
    ds = datasets.MNIST(
        root=str(DATA_DIR),
        train=True,
        download=True,
        transform=transforms.ToTensor(),
    )
    X = torch.stack([ds[i][0] for i in range(len(ds))])  # (60000, 1, 28, 28)
    y = torch.tensor([ds[i][1] for i in range(len(ds))])
    torch.save({"X": X, "y": y}, full_cache)
    return X, y


def load_mnist_7x7(n: int = 500, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (X, y) with X.shape=(n, 49), y.shape=(n,).  Normalized to [0,1]-ish."""
    DATA_DIR.mkdir(exist_ok=True)
    cache = DATA_DIR / f"mnist_7x7_n{n}_seed{seed}.pt"
    if cache.exists():
        blob = torch.load(cache)
        return blob["X"], blob["y"]
    X_full, y_full = _download_full_mnist()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(len(X_full), generator=g)[:n]
    X = X_full[idx]
    y = y_full[idx]
    X = F.avg_pool2d(X, kernel_size=4).reshape(n, 49)
    mean, std = X.mean(), X.std()
    X = (X - mean) / (std + 1e-6)
    torch.save({"X": X, "y": y}, cache)
    return X, y
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_data.py -v`
Expected: 3 passed (may take ~30 s on first run due to MNIST download).

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat(src): cached 7x7 MNIST loader"
```

---

### Task B3: Implement `plotting.py`

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/src/plotting.py`

- [ ] **Step 1: Implement plotting helpers**

```python
"""Shared matplotlib style + helpers for 'money plots' across notebooks."""
import matplotlib.pyplot as plt

STYLE = {
    "figure.figsize": (7.0, 4.0),
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "lines.linewidth": 1.6,
    "font.size": 11,
    "legend.frameon": False,
}


def apply_style() -> None:
    """Call this in the first cell of every notebook."""
    plt.rcParams.update(STYLE)


def semilog_convergence(errors, label=None, ax=None, **kw):
    """Plot convergence on semilog-y."""
    ax = ax or plt.gca()
    ax.semilogy(range(len(errors)), errors, label=label, **kw)
    ax.set_xlabel("iteration")
    ax.set_ylabel("error")
    if label:
        ax.legend()
    return ax


def eigenvalue_compare(true_eigs, ritz_eigs, ax=None):
    """Plot top-k true eigenvalues vs Ritz values, sorted descending."""
    import numpy as np
    ax = ax or plt.gca()
    t = np.sort(np.asarray(true_eigs))[::-1]
    r = np.sort(np.asarray(ritz_eigs))[::-1]
    k = min(len(t), len(r))
    ax.plot(range(k), t[:k], "o-", label="true")
    ax.plot(range(k), r[:k], "x--", label="Ritz")
    ax.set_xlabel("index")
    ax.set_ylabel("eigenvalue")
    ax.legend()
    return ax
```

- [ ] **Step 2: Smoke-test the imports**

Run: `uv run python -c "from src.plotting import apply_style; apply_style(); print('ok')"`
Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add src/plotting.py
git commit -m "feat(src): matplotlib style + money-plot helpers"
```

---

### Task B4: Implement `tests.py` (inline assert harness for notebooks)

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/src/tests.py`

- [ ] **Step 1: Implement the harness**

```python
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
    """Verify reader's power iteration on a symmetric dense matrix.

    your_power signature: your_power(matvec, dim, num_iters, seed) -> (eigval, eigvec)
    """
    from solutions import _01_krylov as sol
    torch.manual_seed(0)
    A = torch.randn(40, 40)
    A = A + A.T
    eigs = torch.linalg.eigvalsh(A)
    top_true = eigs.abs().max().item()

    def matvec(v):
        return A @ v

    eigval, eigvec = your_power(matvec, dim=40, num_iters=200, seed=0)
    assert abs(abs(eigval) - top_true) < 1e-3 * top_true, (
        f"power iter got {eigval:.4f}, true top |eig| {top_true:.4f}"
    )
    assert torch.allclose(matvec(eigvec), eigval * eigvec, atol=1e-3)
    print(f"✓ power iteration: top |λ|={eigval:.4f}, true={top_true:.4f}")


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
    assert err < 1e-3, f"top-5 Ritz off by max |Δ|={err:.2e}"
    QTQ = Q.T @ Q
    orth = (QTQ - torch.eye(Q.shape[1])).abs().max().item()
    assert orth < 1e-4, f"Q not orthonormal: max |QᵀQ - I| = {orth:.2e}"
    print(f"✓ Lanczos: top-5 Ritz match true (max |Δ|={err:.2e}), Q orthonormal")
```

- [ ] **Step 2: Smoke-test imports compile**

Run: `uv run python -c "from src import tests; print('ok')"`
Expected: prints `ok` (it's fine that calling `test_hvp` would fail since solutions aren't written yet).

- [ ] **Step 3: Commit**

```bash
git add src/tests.py
git commit -m "feat(src): inline test harness for notebook exercises"
```

---

## Phase C — Reference implementations (`solutions/01_krylov.py`)

### Task C1: Reference HVP (three variants)

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/solutions/_01_krylov.py`
- Create: `tutorials/numerical-linalg-for-ml/tests/test_hvp.py`

(Note: leading underscore on the solutions module name avoids conflicting with the notebook filename in import paths.)

- [ ] **Step 1: Write failing test**

`tests/test_hvp.py`:

```python
import torch

from solutions._01_krylov import (
    flat_params,
    hvp_finite_difference,
    hvp_double_backward,
    hvp_jvp_of_grad,
    reference_hvp,
)
from src.tiny_models import toy_mlp


def _setup():
    torch.manual_seed(0)
    model = toy_mlp(seed=1)
    X = torch.randn(8, 20)
    y = torch.randint(0, 4, (8,))
    P = sum(p.numel() for p in model.parameters())
    v = torch.randn(P)
    return model, X, y, v


def test_all_three_hvp_agree():
    model, X, y, v = _setup()
    h1 = hvp_double_backward(model, X, y, v)
    h2 = hvp_jvp_of_grad(model, X, y, v)
    h3 = hvp_finite_difference(model, X, y, v, eps=1e-3)
    assert torch.allclose(h1, h2, atol=1e-5)
    assert torch.allclose(h1, h3, atol=5e-3)


def test_reference_hvp_matches_explicit_hessian():
    from torch.func import functional_call
    model, X, y, v = _setup()
    h_ref = reference_hvp(model, X, y, v)

    params_named = {n: p for n, p in model.named_parameters()}

    def loss_fn(theta_flat):
        pd, i = {}, 0
        for n, p in params_named.items():
            pd[n] = theta_flat[i : i + p.numel()].view_as(p)
            i += p.numel()
        return torch.nn.functional.cross_entropy(functional_call(model, pd, (X,)), y)

    theta = flat_params(model)
    H = torch.autograd.functional.hessian(loss_fn, theta)
    h_explicit = H @ v
    assert torch.allclose(h_ref, h_explicit, atol=1e-4)


def test_flat_params_roundtrip():
    model = toy_mlp(seed=2)
    flat = flat_params(model)
    P = sum(p.numel() for p in model.parameters())
    assert flat.shape == (P,)
```

- [ ] **Step 2: Run, verify failure**

Run: `uv run pytest tests/test_hvp.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Implement `solutions/_01_krylov.py` HVP section**

```python
"""Reference implementations for Notebook 1 (Krylov methods).

These are the source of truth that the notebook's exercises check against.
Later notebooks also import from here for ground-truth HVPs.
"""
import torch
import torch.nn.functional as F
from torch.func import functional_call, grad, jvp


def flat_params(model: torch.nn.Module) -> torch.Tensor:
    """Return a single 1D tensor concatenating all parameters."""
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def _params_dict(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {n: p.detach() for n, p in model.named_parameters()}


def _unflatten(flat: torch.Tensor, ref: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out, i = {}, 0
    for n, p in ref.items():
        out[n] = flat[i : i + p.numel()].view_as(p)
        i += p.numel()
    return out


def _loss(params_dict, model, X, y):
    logits = functional_call(model, params_dict, (X,))
    return F.cross_entropy(logits, y)


def hvp_double_backward(model, X, y, v):
    """HVP via double backward.  Conceptually simplest, fp behaves well."""
    params = {n: p for n, p in model.named_parameters()}
    flat = torch.cat([p.reshape(-1) for p in params.values()]).requires_grad_(True)
    pd = _unflatten(flat, params)
    loss = _loss(pd, model, X, y)
    g_flat = torch.autograd.grad(loss, flat, create_graph=True)[0]
    hv = torch.autograd.grad(g_flat @ v, flat, retain_graph=False)[0]
    return hv.detach()


def hvp_jvp_of_grad(model, X, y, v):
    """HVP via JVP of grad — single forward+backward via torch.func."""
    params = _params_dict(model)

    def flat_grad(flat):
        pd = _unflatten(flat, params)
        g_dict = grad(_loss)(pd, model, X, y)
        return torch.cat([g_dict[n].reshape(-1) for n in params])

    flat = flat_params(model)
    _, hvp_flat = jvp(flat_grad, (flat,), (v,))
    return hvp_flat


def hvp_finite_difference(model, X, y, v, eps: float = 1e-3):
    """HVP by central differences on the gradient.  Slow but reference-of-references."""
    params = list(model.parameters())
    P = sum(p.numel() for p in params)

    def grad_at(direction_flat):
        # Restore original after.
        with torch.no_grad():
            i = 0
            for p in params:
                p.add_(direction_flat[i : i + p.numel()].view_as(p))
                i += p.numel()
        loss = F.cross_entropy(model(X), y)
        g = torch.autograd.grad(loss, list(model.parameters()))
        g_flat = torch.cat([gi.reshape(-1) for gi in g])
        with torch.no_grad():
            i = 0
            for p in params:
                p.sub_(direction_flat[i : i + p.numel()].view_as(p))
                i += p.numel()
        return g_flat

    g_plus = grad_at(eps * v)
    g_minus = grad_at(-eps * v)
    return (g_plus - g_minus) / (2 * eps)


reference_hvp = hvp_double_backward
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_hvp.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add solutions/_01_krylov.py tests/test_hvp.py
git commit -m "feat(solutions): reference HVP (3 variants) for Notebook 1"
```

---

### Task C2: Reference power iteration with deflation

**Files:**
- Modify: `tutorials/numerical-linalg-for-ml/solutions/_01_krylov.py`
- Create: `tutorials/numerical-linalg-for-ml/tests/test_power.py`

- [ ] **Step 1: Write failing test**

`tests/test_power.py`:

```python
import torch

from solutions._01_krylov import power_iteration, power_iteration_deflated


def test_power_iteration_top_eigval():
    torch.manual_seed(0)
    A = torch.diag(torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5]))
    Q = torch.linalg.qr(torch.randn(5, 5))[0]
    A = Q @ A @ Q.T  # spectrum {5,3,2,1,0.5} in a non-canonical basis

    def matvec(v):
        return A @ v

    eigval, vec, history = power_iteration(matvec, dim=5, num_iters=200, seed=0)
    assert abs(eigval - 5.0) < 1e-4
    # convergence history should be monotone non-increasing (mostly)
    assert history[-1] < history[10]


def test_deflation_top3():
    torch.manual_seed(0)
    A = torch.diag(torch.tensor([5.0, 3.0, 2.0, 1.0, 0.5]))
    Q = torch.linalg.qr(torch.randn(5, 5))[0]
    A = Q @ A @ Q.T

    def matvec(v):
        return A @ v

    eigvals, _ = power_iteration_deflated(
        matvec, dim=5, k=3, num_iters_per=400, seed=0
    )
    expected = torch.tensor([5.0, 3.0, 2.0])
    assert torch.allclose(torch.tensor(eigvals), expected, atol=1e-3)
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_power.py -v`
Expected: FAIL (functions don't exist yet).

- [ ] **Step 3: Append to `solutions/_01_krylov.py`**

```python
def power_iteration(matvec, dim, num_iters=200, tol=1e-10, seed=0):
    """Plain power iteration.

    Returns: (top eigenvalue estimate, top eigvec, list of per-iter Rayleigh
    quotient errors vs final estimate).
    """
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(dim, generator=g)
    v = v / v.norm()
    history = []
    eigval = 0.0
    for _ in range(num_iters):
        Av = matvec(v)
        new_eigval = (v @ Av).item()
        v = Av / (Av.norm() + 1e-30)
        history.append(abs(new_eigval - eigval))
        if abs(new_eigval - eigval) < tol:
            eigval = new_eigval
            break
        eigval = new_eigval
    return eigval, v, history


def power_iteration_deflated(matvec, dim, k, num_iters_per=300, seed=0):
    """Power iteration with rank-1 deflation to recover top-k eigenpairs."""
    g = torch.Generator().manual_seed(seed)
    found_vals, found_vecs = [], []

    def deflated_matvec(v):
        out = matvec(v)
        for lam, u in zip(found_vals, found_vecs):
            out = out - lam * (u @ v) * u
        return out

    for _ in range(k):
        v0 = torch.randn(dim, generator=g)
        v0 = v0 / v0.norm()
        # Reuse power_iteration logic, but with the deflated operator.
        v = v0
        eigval = 0.0
        for _ in range(num_iters_per):
            Av = deflated_matvec(v)
            new_eigval = (v @ Av).item()
            v = Av / (Av.norm() + 1e-30)
            eigval = new_eigval
        found_vals.append(eigval)
        found_vecs.append(v)
    return found_vals, found_vecs
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_power.py -v`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add solutions/_01_krylov.py tests/test_power.py
git commit -m "feat(solutions): power iteration + rank-1 deflation"
```

---

### Task C3: Reference Lanczos (none / full / selective reorth)

**Files:**
- Modify: `tutorials/numerical-linalg-for-ml/solutions/_01_krylov.py`
- Create: `tutorials/numerical-linalg-for-ml/tests/test_lanczos.py`

- [ ] **Step 1: Write failing test**

`tests/test_lanczos.py`:

```python
import torch

from solutions._01_krylov import lanczos


def _random_symmetric(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(n, n, generator=g)
    return A + A.T


def test_lanczos_no_reorth_runs():
    A = _random_symmetric(40)
    ritz, Q = lanczos(lambda v: A @ v, dim=40, k=20, reorth="none")
    assert ritz.shape == (20,)
    assert Q.shape == (40, 20)


def test_lanczos_full_reorth_orthonormal():
    A = _random_symmetric(40)
    ritz, Q = lanczos(lambda v: A @ v, dim=40, k=20, reorth="full")
    QTQ = Q.T @ Q
    err = (QTQ - torch.eye(20)).abs().max().item()
    assert err < 1e-5, f"Q not orthonormal: max |QᵀQ - I| = {err:.2e}"


def test_lanczos_full_reorth_top_eigs():
    A = _random_symmetric(60, seed=42)
    true_eigs = torch.linalg.eigvalsh(A)
    top3 = true_eigs[-3:].flip(0)
    ritz, _ = lanczos(lambda v: A @ v, dim=60, k=30, reorth="full")
    top3_ritz = ritz.sort(descending=True).values[:3]
    assert torch.allclose(top3_ritz, top3, atol=1e-4)


def test_lanczos_selective_orthogonality():
    """Selective reorth should keep ||QᵀQ - I|| below 1e-3 over 30 steps."""
    A = _random_symmetric(40)
    _, Q = lanczos(lambda v: A @ v, dim=40, k=30, reorth="selective")
    QTQ = Q.T @ Q
    err = (QTQ - torch.eye(30)).abs().max().item()
    assert err < 1e-3, f"selective reorth failed: |QᵀQ-I|={err:.2e}"


def test_lanczos_no_reorth_loses_orthogonality_eventually():
    """Without reorth, we lose orthogonality by step ~25 on a hard matrix."""
    A = _random_symmetric(40, seed=7).float()
    _, Q = lanczos(lambda v: A @ v, dim=40, k=35, reorth="none")
    QTQ = Q.T @ Q
    err = (QTQ - torch.eye(35)).abs().max().item()
    assert err > 1e-3, f"expected orthogonality loss but err={err:.2e}"
```

- [ ] **Step 2: Run, verify fail**

Run: `uv run pytest tests/test_lanczos.py -v`
Expected: FAIL.

- [ ] **Step 3: Append Lanczos to `solutions/_01_krylov.py`**

```python
def lanczos(matvec, dim, k, reorth: str = "selective", seed: int = 0):
    """Lanczos tridiagonalization with three reorthogonalization modes.

    Args:
        matvec: callable v -> Av, for symmetric A.
        dim: ambient dimension.
        k: number of Lanczos steps (=> at most k Ritz values).
        reorth: 'none', 'full', or 'selective'.
        seed: RNG seed for the starting vector.

    Returns: (ritz_values, Q) where Q is (dim, k) with columns the Lanczos basis.
    """
    if reorth not in {"none", "full", "selective"}:
        raise ValueError(f"reorth={reorth!r}")
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=g)
    q = q / q.norm()
    Q = torch.zeros(dim, k)
    alphas = torch.zeros(k)
    betas = torch.zeros(k - 1)
    q_prev = torch.zeros(dim)
    beta_prev = 0.0

    for j in range(k):
        Q[:, j] = q
        Aq = matvec(q)
        alpha = q @ Aq
        alphas[j] = alpha
        r = Aq - alpha * q - beta_prev * q_prev

        if reorth == "full" and j > 0:
            # Full reorthogonalization against all previous basis vectors.
            r = r - Q[:, : j + 1] @ (Q[:, : j + 1].T @ r)
            r = r - Q[:, : j + 1] @ (Q[:, : j + 1].T @ r)  # twice-is-enough
        elif reorth == "selective" and j > 0:
            # Reorth only if ||r|| dropped a lot (Paige-style estimate).
            r_norm = r.norm()
            if r_norm < 0.717 * (Aq.norm()):
                r = r - Q[:, : j + 1] @ (Q[:, : j + 1].T @ r)

        beta = r.norm().item()
        if j < k - 1:
            betas[j] = beta
            if beta < 1e-14:
                # Lucky breakdown: invariant subspace.  Pad with zeros.
                break
            q_prev = q
            q = r / beta
            beta_prev = beta

    T = torch.diag(alphas) + torch.diag(betas, 1) + torch.diag(betas, -1)
    ritz = torch.linalg.eigvalsh(T)
    return ritz, Q
```

- [ ] **Step 4: Run tests, verify pass**

Run: `uv run pytest tests/test_lanczos.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add solutions/_01_krylov.py tests/test_lanczos.py
git commit -m "feat(solutions): Lanczos with none/full/selective reorth"
```

---

### Task C4: Update `src/tests.py` to add Lanczos selective-reorth check

(Done in earlier step B4 — verify it now passes.)

- [ ] **Step 1: Run the harness against the references**

Run:

```bash
uv run python -c "
from src.tests import test_hvp, test_power_iteration, test_lanczos
from solutions._01_krylov import (
    reference_hvp, power_iteration, lanczos,
)
test_hvp(reference_hvp)
test_power_iteration(power_iteration)
test_lanczos(lambda mv, dim, k, reorth='selective', seed=0:
             lanczos(mv, dim, k, reorth=reorth, seed=seed))
"
```

Expected: three `✓` lines printed.

- [ ] **Step 2: Commit (no file changes, just confirmation)**

(Nothing to commit; this task confirms the harness wires up.)

---

## Phase D — Notebook 1 build script

### Task D1: Notebook builder skeleton

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/notebooks/build_01.py`

- [ ] **Step 1: Implement the builder skeleton**

```python
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


# === Section stubs (to be filled in next tasks) ===

def _section_preamble(cells):
    cells.append(md("# Notebook 1 — Krylov methods for the Hessian\n\n_(preamble replaced in Task D2)_"))


def _section_0_nla_prelim(cells):
    pass


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
```

- [ ] **Step 2: Run the builder, verify it produces a (mostly empty) notebook**

Run: `cd ~/Programming/Claude/tutorials/numerical-linalg-for-ml && uv run python notebooks/build_01.py`
Expected: prints `wrote notebooks/01_krylov.ipynb (1 cells)`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(notebooks): scaffold build_01.py + initial 01_krylov.ipynb"
```

---

### Task D2: Section preamble

**Files:**
- Modify: `notebooks/build_01.py:_section_preamble`

- [ ] **Step 1: Replace `_section_preamble`**

```python
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
```

- [ ] **Step 2: Rebuild and inspect**

Run: `uv run python notebooks/build_01.py`
Expected: `wrote notebooks/01_krylov.ipynb (2 cells)`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(nb1): preamble + imports cell"
```

---

### Task D3: Section 0 — NLA preliminaries

**Files:**
- Modify: `notebooks/build_01.py:_section_0_nla_prelim`

- [ ] **Step 1: Replace `_section_0_nla_prelim`**

```python
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
    cells.append(code("""
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
plt.xlabel(r'$\\kappa(A)$'); plt.ylabel('relative residual'); plt.title('fp32 solve precision vs condition number')
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
```

- [ ] **Step 2: Rebuild and verify**

Run: `uv run python notebooks/build_01.py`
Expected: `wrote notebooks/01_krylov.ipynb (10 cells)`.

- [ ] **Step 3: Execute the notebook to make sure all section-0 cells run**

Run: `uv run jupyter nbconvert --to notebook --execute notebooks/01_krylov.ipynb --output 01_krylov_check.ipynb`
Expected: success (it will fail at the first `YOUR CODE HERE` since `assert ... is not None` triggers; we need to skip-execute the exercise cells).

Actually: execution will halt at exercise asserts.  Instead verify by parsing only:

Run: `uv run python -c "
import nbformat
nb = nbformat.read('notebooks/01_krylov.ipynb', as_version=4)
print(f'{len(nb.cells)} cells, types:', [c.cell_type for c in nb.cells])
"`
Expected: 10 cells with alternating markdown/code.

- [ ] **Step 4: Commit**

```bash
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(nb1): section 0 — NLA preliminaries (norms, kappa, eps, Rayleigh)"
```

---

### Task D4: Section 1 — HVP three ways

**Files:**
- Modify: `notebooks/build_01.py:_section_1_hvp`

- [ ] **Step 1: Replace `_section_1_hvp`**

```python
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
```

- [ ] **Step 2: Rebuild and verify**

Run: `uv run python notebooks/build_01.py`
Expected: `wrote notebooks/01_krylov.ipynb (~20 cells)`.

- [ ] **Step 3: Commit**

```bash
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(nb1): section 1 — HVP three ways"
```

---

### Task D5: Section 2 — Power iteration

**Files:**
- Modify: `notebooks/build_01.py:_section_2_power`

- [ ] **Step 1: Replace `_section_2_power`**

```python
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
    cells.append(code("""
def make_diag_spd(eigs):
    return torch.diag(torch.tensor(eigs, dtype=torch.float64))

specs = {
    'wide gap (λ₂/λ₁ = 0.1)':   [10.0] + [1.0]*49,
    'narrow gap (λ₂/λ₁ = 0.95)': [10.0, 9.5] + [1.0]*48,
    'tied (λ₂/λ₁ = 1.0)':        [10.0, 10.0] + [1.0]*48,
}

fig, ax = plt.subplots()
for name, eigs in specs.items():
    A = make_diag_spd(eigs)
    matvec = lambda v: A @ v.double()
    _, _, hist = power_iteration(matvec, dim=50, num_iters=120, seed=0)
    ax.semilogy(hist, label=name)
ax.set_xlabel('iteration'); ax.set_ylabel(r'$|\\lambda^{(k)} - \\lambda^{(k-1)}|$')
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
```

- [ ] **Step 2: Rebuild and commit**

```bash
uv run python notebooks/build_01.py
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(nb1): section 2 — power iteration + convergence plot"
```

---

### Task D6: Section 3 — Deflation

**Files:**
- Modify: `notebooks/build_01.py:_section_3_deflation`

- [ ] **Step 1: Replace `_section_3_deflation`**

```python
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
    cells.append(code("""
def power_iteration_deflated(matvec, dim, k, num_iters_per=300, seed=0):
    # YOUR CODE HERE
    raise NotImplementedError

eigvals, eigvecs = power_iteration_deflated(matvec_H, dim=P, k=3, num_iters_per=400, seed=0)
# Compare by magnitude — power iteration finds largest-|λ|, and the Hessian
# may be indefinite at this point in training.
top3_by_mag = true_eigs.abs().sort(descending=True).values[:3]
print(f'top-3 by power+deflation (signed):    {[f\"{x:+.3f}\" for x in eigvals]}')
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
    cells.append(code("""
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
```

- [ ] **Step 2: Rebuild and commit**

```bash
uv run python notebooks/build_01.py
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(nb1): section 3 — deflation + degradation plot"
```

---

### Task D7: Section 4 — Lanczos three-term recurrence

**Files:**
- Modify: `notebooks/build_01.py:_section_4_lanczos`

- [ ] **Step 1: Replace `_section_4_lanczos`**

```python
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
```

- [ ] **Step 2: Rebuild and commit**

```bash
uv run python notebooks/build_01.py
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(nb1): section 4 — Lanczos three-term recurrence + Ritz convergence"
```

---

### Task D8: Section 5 — Loss of orthogonality

**Files:**
- Modify: `notebooks/build_01.py:_section_5_orthogonality`

- [ ] **Step 1: Replace `_section_5_orthogonality`**

```python
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
    cells.append(code("""
def lanczos_track_orth(matvec, dim, k, reorth='none', seed=0):
    \"\"\"As above but also returns orthogonality-error history.\"\"\"
    # YOUR CODE HERE: copy your lanczos_no_reorth and add per-step orth tracking.
    raise NotImplementedError

A40 = (torch.randn(40, 40, generator=torch.Generator().manual_seed(7))).float()
A40 = A40 + A40.T

_, _, orth_none = lanczos_track_orth(lambda v: A40 @ v, dim=40, k=40, reorth='none')

plt.figure()
plt.semilogy(orth_none, label='no reorth (fp32)')
plt.xlabel('step'); plt.ylabel(r'$\\| Q^\\top Q - I \\|_\\infty$')
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
    cells.append(code("""
_, _, orth_full = lanczos_track_orth(lambda v: A40 @ v, dim=40, k=40, reorth='full')

plt.figure()
plt.semilogy(orth_none, label='no reorth')
plt.semilogy(orth_full, label='full reorth (twice)')
plt.xlabel('step'); plt.ylabel(r'$\\| Q^\\top Q - I \\|_\\infty$')
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
    cells.append(code("""
_, _, orth_sel = lanczos_track_orth(lambda v: A40 @ v, dim=40, k=40, reorth='selective')

plt.figure(figsize=(8, 4.5))
plt.semilogy(orth_none, label='none')
plt.semilogy(orth_full, label='full (twice-is-enough)')
plt.semilogy(orth_sel,  label='selective (Paige)')
plt.axhline(1e-7, color='k', linestyle=':', alpha=0.6, label='fp32 eps')
plt.xlabel('Lanczos step'); plt.ylabel(r'$\\| Q^\\top Q - I \\|_\\infty$')
plt.title('Money plot: orthogonality loss across reorth strategies')
plt.legend(); plt.show()
"""))
```

- [ ] **Step 2: Rebuild and commit**

```bash
uv run python notebooks/build_01.py
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(nb1): section 5 — orthogonality loss + reorth strategies (money plot)"
```

---

### Task D9: Section 6 — Hessian top-k in practice

**Files:**
- Modify: `notebooks/build_01.py:_section_6_hessian_topk`

- [ ] **Step 1: Replace `_section_6_hessian_topk`**

```python
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
    cells.append(code("""
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
ax.set_xlabel('rank'); ax.set_ylabel(r'$|\\lambda_k|$')
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
```

- [ ] **Step 2: Rebuild and commit**

```bash
uv run python notebooks/build_01.py
git add notebooks/build_01.py notebooks/01_krylov.ipynb
git commit -m "feat(nb1): section 6 — Hessian top-k in practice + wrap-up"
```

---

## Phase E — Notebook 1 end-to-end smoke test + README

### Task E1: End-to-end execute Notebook 1 with solutions injected

**Files:**
- Create: `notebooks/_run_with_solutions.py`

This script swaps every `raise NotImplementedError` for the corresponding
function from `solutions/_01_krylov.py`, then executes the notebook.  This is
how we know the notebook actually runs.

- [ ] **Step 1: Create the swap-and-run script**

```python
"""Inject solutions into 01_krylov.ipynb and execute it.

Used as the CI smoke test for Notebook 1.  Run:
    uv run python notebooks/_run_with_solutions.py
"""
import re
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).parent / "01_krylov.ipynb"

REPLACEMENTS = [
    # (function name in notebook, function name to import from solutions)
    ("hvp_double_backward",     "hvp_double_backward"),
    ("hvp_jvp_of_grad",         "hvp_jvp_of_grad"),
    ("hvp_finite_difference",   "hvp_finite_difference"),
    ("power_iteration",         "power_iteration"),
    ("power_iteration_deflated","power_iteration_deflated"),
    ("lanczos_no_reorth",       "lanczos"),
    ("lanczos_track_orth",      "lanczos_track_orth"),
]

INJECTION = """
# === INJECTED BY _run_with_solutions.py ===
from solutions._01_krylov import (
    hvp_double_backward as _ref_hvp_double,
    hvp_jvp_of_grad     as _ref_hvp_jvp,
    hvp_finite_difference as _ref_hvp_fd,
    power_iteration,
    power_iteration_deflated,
    lanczos,
)
hvp_double_backward = _ref_hvp_double
hvp_jvp_of_grad = _ref_hvp_jvp
hvp_finite_difference = _ref_hvp_fd
def lanczos_no_reorth(matvec, dim, k, seed=0):
    ritz, Q = lanczos(matvec, dim=dim, k=k, reorth='none', seed=seed)
    return ritz, Q
def lanczos_track_orth(matvec, dim, k, reorth='none', seed=0):
    # Re-run lanczos but also track orthogonality.  Quick standalone impl.
    import torch
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(dim, generator=g); q = q / q.norm()
    Q = torch.zeros(dim, k); alphas = torch.zeros(k); betas = torch.zeros(k-1)
    q_prev = torch.zeros(dim); beta_prev = 0.0; orth_hist = []
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
        orth_hist.append((Q[:, :j+1].T @ Q[:, :j+1] - torch.eye(j+1)).abs().max().item())
    T = torch.diag(alphas) + torch.diag(betas, 1) + torch.diag(betas, -1)
    return torch.linalg.eigvalsh(T), Q, orth_hist
# === END INJECTION ===
"""


def main():
    nb = nbformat.read(NB, as_version=4)
    # Replace every `raise NotImplementedError` cell body by appending the injection,
    # which shadows the reader's (empty) functions with the references.
    for cell in nb.cells:
        if cell.cell_type == "code" and "raise NotImplementedError" in cell.source:
            cell.source = cell.source + "\n" + INJECTION
    client = NotebookClient(nb, timeout=120, kernel_name="python3")
    client.execute()
    print("✓ notebook executed end-to-end")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Add `nbclient` to dev deps**

Edit `pyproject.toml`:

```toml
[dependency-groups]
dev = [
    "pytest>=8.0",
    "ruff>=0.5",
    "nbclient>=0.10",
    "ipykernel>=6.29",
]
```

Run: `uv sync`

- [ ] **Step 3: Run the smoke test**

Run: `cd ~/Programming/Claude/tutorials/numerical-linalg-for-ml && uv run python notebooks/_run_with_solutions.py`
Expected: `✓ notebook executed end-to-end` (may take 1-2 min).

If it fails, the failure message points to the cell.  Common fixes: missing
imports in the injection, function-signature drift between notebook and
solutions.

- [ ] **Step 4: Commit**

```bash
git add notebooks/_run_with_solutions.py pyproject.toml uv.lock
git commit -m "test(nb1): end-to-end smoke test via injected solutions"
```

---

### Task E2: Write the README

**Files:**
- Create: `tutorials/numerical-linalg-for-ml/README.md`

- [ ] **Step 1: Write README**

```markdown
# Numerical Linear Algebra for ML — Hands-On Survey

ARENA-style Jupyter exercises in matrix-free numerical linear algebra,
focused on the two objects ML researchers actually care about: the **loss
Hessian** and the **empirical NTK**.

## Status

| Notebook                        | Status      | Time   |
|---------------------------------|-------------|--------|
| 1. Krylov (power iter + Lanczos)| ✅ shipped  | ~145m  |
| 2. Randomized (rSVD + eNTK)     | ⏳ planned  | ~100m  |
| 3. Estimation + Perturbation    | ⏳ planned  | ~115m  |
| 4. Capstone (CNN spectroscopy)  | ⏳ planned  | ~120m  |

See `design.md` for the full design.

## Getting started

```bash
cd ~/Programming/Claude/tutorials/numerical-linalg-for-ml
uv sync
uv run jupyter lab notebooks/01_krylov.ipynb
```

Solutions live in `solutions/`.  Don't peek until you've tried.

## The O() table

| Operation                        | Cost                                        |
|----------------------------------|---------------------------------------------|
| matvec (Hessian or eNTK)         | ≈ O(forward + backward) ≈ O(P)              |
| power iteration, top-1, ε err    | k = O(log(1/ε) / log(λ₁/λ₂)) matvecs        |
| Lanczos, top-k extremal eigvecs  | O(k) matvecs + O(k²) for reorth             |
| randomized SVD, rank-k + p over  | O((k+p) matvecs) + O(N(k+p)²) small SVD     |
| Hutchinson trace, m probes       | O(m) matvecs; Var = 2‖A‖_F²/m (Gaussian)    |
| stochastic Lanczos quadrature    | O(m probes · s Lanczos steps) matvecs       |
| eNTK matvec (Novak et al.)       | O(forward + backward), NOT O(N²P)            |

**Stability bounds (symmetric A, perturbation E):**

| Bound        | Statement                                    |
|--------------|----------------------------------------------|
| Weyl         | |Δλ_k| ≤ ‖E‖₂                                |
| Davis-Kahan  | sin Θ_k ≲ ‖E‖₂ / gap_k                       |
| stable rank  | r_s(A) = ‖A‖_F² / ‖A‖_2² (small → rSVD wins) |

## Money plots

| Notebook | The plot you'd put in a paper                                   |
|----------|------------------------------------------------------------------|
| 1        | Orthogonality loss in classical Lanczos: none / full / selective  |
| 2        | Matrix-free eNTK matvec time stays flat; materialized explodes    |
| 3        | Hessian DOS: bulk near zero + a handful of outliers (log y)       |
| 4        | DOS across training: bulk consolidates, outliers separate         |

## Repo layout

```
src/         shared models, plotting, data loaders, test harness
solutions/   reference implementations (also notebook-1's solutions)
notebooks/   .ipynb files + build_*.py scripts that generate them
tests/       pytest unit tests for solutions/ and src/
data/        cached MNIST/CIFAR subsets (not committed)
```

## Development

Notebooks are built from `notebooks/build_*.py` scripts.  Editing the `.py`
is the source-of-truth path; the `.ipynb` is regenerated.

```bash
# rebuild a notebook from its build script
uv run python notebooks/build_01.py

# run the unit tests
uv run pytest

# end-to-end smoke test of a notebook
uv run python notebooks/_run_with_solutions.py
```
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: README with status, O() table, money plots, how to run"
```

---

### Task E3: Final sanity sweep

- [ ] **Step 1: Run full test suite**

Run: `uv run pytest -v`
Expected: all tests pass (test_models, test_data, test_hvp, test_power, test_lanczos).

- [ ] **Step 2: Rebuild and re-execute Notebook 1**

Run:

```bash
uv run python notebooks/build_01.py
uv run python notebooks/_run_with_solutions.py
```

Expected: notebook builds and executes end-to-end.

- [ ] **Step 3: List final tree and commit any stray cleanups**

Run: `find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './data/*' -not -path './__pycache__/*' -not -path './*/__pycache__/*' | sort`

Expected (give or take):
```
./.gitignore
./README.md
./design.md
./notebooks/01_krylov.ipynb
./notebooks/_run_with_solutions.py
./notebooks/build_01.py
./plan-01-scaffolding-and-krylov.md
./pyproject.toml
./solutions/__init__.py
./solutions/_01_krylov.py
./src/__init__.py
./src/data.py
./src/plotting.py
./src/tests.py
./src/tiny_models.py
./tests/test_data.py
./tests/test_hvp.py
./tests/test_lanczos.py
./tests/test_models.py
./tests/test_power.py
./uv.lock
```

- [ ] **Step 4: Done**

Phase 1 complete.  Notebook 1 is shippable.  Next: plan-02 for Notebook 2.

---

## Spec coverage check

| Design spec item                         | Plan task               |
|-------------------------------------------|-------------------------|
| Project scaffolding (uv, dirs)            | A1, A2                  |
| `toy_mlp` (~250p), `tiny_mlp` (~2k)       | B1                      |
| MNIST 7×7 cached loader                   | B2                      |
| Plotting helpers + shared style           | B3                      |
| Inline `src/tests.py` harness             | B4                      |
| Reference HVP (3 variants)                | C1                      |
| Reference power iteration + deflation     | C2                      |
| Reference Lanczos (none/full/selective)   | C3                      |
| Notebook 1 §0: NLA preliminaries          | D3                      |
| Notebook 1 §1: HVP three ways             | D4                      |
| Notebook 1 §2: Power iteration            | D5                      |
| Notebook 1 §3: Deflation                  | D6                      |
| Notebook 1 §4: Lanczos three-term         | D7                      |
| Notebook 1 §5: Orthogonality + reorth     | D8                      |
| Notebook 1 §6: Hessian top-k in practice  | D9                      |
| Notebook 1 end-to-end smoke test          | E1                      |
| README with O() table + money plots       | E2                      |
| Final sanity sweep                        | E3                      |

(Deferred to Plan 02-04: Notebooks 2, 3, 4; eNTK matvec; rSVD; Hutchinson;
SLQ; perturbation theory; capstone CNN.  Plan 02 onward will build on the
infrastructure committed in this phase.)
