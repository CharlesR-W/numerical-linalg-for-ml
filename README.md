# Numerical Linear Algebra for ML — Hands-On Survey

ARENA-style Jupyter exercises in matrix-free numerical linear algebra,
focused on the two objects ML researchers actually care about: the **loss
Hessian** and the **empirical NTK**.

The angle: the matrices we care about are too large to materialize but
cheap to apply.  So the question is never "how do I decompose this dense
matrix" but "what can I learn from black-box matvecs?"  The answer is
Krylov + randomization + stochastic estimation, in a few hundred lines
of PyTorch.

Companion blog post: [Numerical Linear Algebra for ML](https://charlesr-w.github.io/crw-blog/tutorials/)

## Notebooks

| Notebook                                | Time   | Topics                                                 |
|-----------------------------------------|--------|--------------------------------------------------------|
| 1. Krylov (power iter + Lanczos)        | ~145m  | HVPs, deflation, Lanczos, loss of orthogonality        |
| 2. Randomized (rSVD + eNTK)             | ~100m  | stable rank, HMT range finder, matrix-free eNTK matvec |
| 3. Estimation + Perturbation            | ~115m  | Hutchinson trace, SLQ density of states, Weyl, Davis-Kahan |
| 4. Capstone (tiny_mlp across training)  | ~120m  | spectroscopy across 4 training checkpoints            |

See `design.md` for the design rationale, scope, and pedagogical choices.

## Getting started

```bash
git clone https://github.com/CharlesR-W/numerical-linalg-for-ml.git
cd numerical-linalg-for-ml
uv sync --all-groups
uv run jupyter lab notebooks/01_krylov.ipynb
```

Exercises are `# YOUR CODE HERE` stubs.  Reference solutions live in
`solutions/` -- don't peek until you've tried.

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
solutions/   reference implementations
notebooks/   .ipynb files + build_*.py scripts that generate them
tests/       pytest unit tests for solutions/ and src/
scripts/     training script that produces the capstone checkpoints
data/        cached MNIST subset + capstone checkpoints
```

## Development

Notebooks are built from `notebooks/build_*.py` scripts.  Editing the
`.py` is the source-of-truth path; the `.ipynb` is regenerated.

```bash
# rebuild a notebook from its build script
uv run python notebooks/build_01.py

# run the unit tests
uv run pytest

# end-to-end smoke test of a notebook (injects solutions, executes all cells)
uv run python notebooks/_run_with_solutions.py
```

## Credits

Designed and curated by Charles Renshaw-Whitman.  Written with Claude
(Anthropic) -- implementation, exercise stubs, and reference solutions.
