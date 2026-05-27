# Numerical Linear Algebra for ML — Hands-On Survey

ARENA-style Jupyter exercises in matrix-free numerical linear algebra,
focused on the two objects ML researchers actually care about: the **loss
Hessian** and the **empirical NTK**.

## Status

| Notebook                          | Status      | Time   |
|-----------------------------------|-------------|--------|
| 1. Krylov (power iter + Lanczos)  | shipped     | ~145m  |
| 2. Randomized (rSVD + eNTK)       | shipped     | ~100m  |
| 3. Estimation + Perturbation      | planned     | ~115m  |
| 4. Capstone (CNN spectroscopy)    | planned     | ~120m  |

See `design.md` for the full design.

## Getting started

```bash
cd ~/Programming/Claude/tutorials/numerical-linalg-for-ml
uv sync --all-groups
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
| eNTK matvec (Novak et al.)       | O(forward + backward), NOT O(N²P)           |

**Stability bounds (symmetric A, perturbation E):**

| Bound        | Statement                                    |
|--------------|----------------------------------------------|
| Weyl         | \|Δλ_k\| ≤ ‖E‖₂                              |
| Davis-Kahan  | sin Θ_k ≲ ‖E‖₂ / gap_k                       |
| stable rank  | r_s(A) = ‖A‖_F² / ‖A‖_2² (small → rSVD wins) |

## Money plots

| Notebook | The plot you'd put in a paper                                  |
|----------|-----------------------------------------------------------------|
| 1        | Orthogonality loss in classical Lanczos: none / full / selective |
| 2        | Matrix-free eNTK matvec time stays flat; materialized explodes   |
| 3        | Hessian DOS: bulk near zero + a handful of outliers (log y)      |
| 4        | DOS across training: bulk consolidates, outliers separate        |

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

# end-to-end smoke test of a notebook (injects solutions, executes all cells)
uv run python notebooks/_run_with_solutions.py
```
