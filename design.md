# Design: Numerical Linear Algebra for ML — A Hands-On Survey

## Goal

A short, ARENA-style Jupyter tutorial that takes an ML researcher from "I
know what an eigenvalue is" to "I can probe the Hessian and eNTK of a real
network and understand the cost of every step."  Four chunky notebooks,
each a 60-120 minute sit-down.  Survey-paper breadth, but with `# YOUR
CODE HERE` exercises throughout so the reader actually builds the methods.

The two anchor objects are the **Hessian** of a loss and the **empirical
NTK** (eNTK) of a network's output map.  Both are too large to materialize
in any setting that matters, and both share a single trick: express the
matvec as a single autodiff sweep over parameters, never form the matrix.

## Audience and assumptions

Readers are ML researchers comfortable with PyTorch and basic autodiff
(forward + reverse mode).  We assume they know what a Hessian is in
principle but have never used Lanczos on one; they may have seen randomized
SVD as a name but not a code path.  **We assume no formal NLA background.**
Notebook 1 opens with a brief preliminaries section (operator and Frobenius
norms, condition number, machine epsilon, Rayleigh quotient) that the rest
of the tutorial leans on.  Eigenvalue perturbation theory (Weyl, Davis-Kahan)
appears in Notebook 3 as the closing "how much do you trust these
estimates?" section.  No measure theory, no proofs of convergence rates —
all bounds are stated and verified empirically.

## Out of scope

- CG / MINRES / GMRES and the rest of iterative linear solves.  Influence
  functions and natural-gradient steps need these; they deserve their own
  followup tutorial.
- Preconditioners and second-order optimizers (K-FAC, Shampoo, Sophia).
- Multi-GPU, sharded, or truly large-scale runs.  Everything must run on a
  laptop CPU in well under a minute per cell.
- Proofs.  This is a numerics-and-intuition tutorial, not a textbook.

## Library and format choices

- **PyTorch + `torch.func`** (`jvp`, `vjp`, `grad`, `vmap`, `hessian`).
  Matches ARENA's convention; the `torch.func` API exposes JVP/VJP cleanly
  enough that the matrix-free story reads well in code.
- **Jupyter `.ipynb` notebooks**, not marimo or Quarto.  ARENA-style means
  the page IS the exercise harness.
- **Solutions** in a sibling `solutions/` folder (one `.py` or `.ipynb` per
  notebook).  The instructions notebook contains stubs + `assert`-based
  tests inline; the solutions file contains the filled-in implementations.
- Difficulty markers (🔴⚪⚪⚪⚪ to 🔴🔴🔴🔴🔴) and time estimates on
  each exercise, ARENA-style.

## Layout

```
tutorials/numerical-linalg-for-ml/
├── README.md          # one-page index, the O() table, the four "money plots"
├── design.md          # this file
├── notebooks/
│   ├── 01_krylov.ipynb       # ~120 min  Hessian + power iter + Lanczos
│   ├── 02_randomized.ipynb   # ~90 min   rSVD + matrix-free eNTK
│   ├── 03_estimation.ipynb   # ~90 min   Hutchinson + SLQ
│   └── 04_capstone.ipynb     # ~120 min  spectroscopy of a small CNN
├── solutions/
│   ├── 01_krylov_solutions.py
│   ├── 02_randomized_solutions.py
│   ├── 03_estimation_solutions.py
│   └── 04_capstone_solutions.py
├── src/
│   ├── tiny_models.py   # a 3-layer MLP and a tiny CNN used everywhere
│   ├── hvp.py           # reference HVP / eNTK matvec implementations
│   ├── plotting.py      # shared matplotlib style, helper plotters
│   └── tests.py         # asserts called from the notebooks
└── data/                # MNIST/CIFAR loaders, cached subsets
```

The `src/` package exists so notebooks stay short and so the same `tiny_mlp()`
appears in every notebook with identical weights when seeded.

## The unifying O() table (lives in README and Notebook 1)

```
matvec cost                       ≈ O(forward+backward) ≈ O(P) for dense net
power iteration, top-1, ε err     k = O(log(1/ε) / log(λ₁/λ₂)) matvecs
Lanczos, top-k extremal eigvecs   O(k) matvecs + O(k²) tridiag eig
randomized SVD, rank-k + p over   O((k+p) matvecs) + O(N(k+p)²) small SVD
Hutchinson trace, m probes        O(m) matvecs;  Var ≈ 2‖A‖_F² / m  (Rademacher)
stochastic Lanczos quadrature     O(m probes · s Lanczos steps) matvecs
eNTK matvec (Novak et al.)        O(forward+backward),   NOT O(N²P)

stability bounds (symmetric A, perturbation E):
  Weyl              |Δλ_k|       ≤  ‖E‖_2
  Davis-Kahan       sin Θ_k      ≲  ‖E‖_2 / gap_k
  stable rank       r_s(A)       =  ‖A‖_F² / ‖A‖_2²       (small → rSVD wins)
```

Every notebook ends by updating a running "where we are on this table" line.

---

## Notebook 1 — Krylov: power iteration and Lanczos, with the Hessian

**Time:** ~145 min.  Seven sections, ~16 small exercises.
**Anchor:** the loss Hessian of a tiny MLP on a few hundred MNIST samples.
**Opens with** a ~25-min NLA preliminaries section that the rest of the
tutorial leans on.

### Sections

0. **NLA you'll need** (~25 min, 4 exercises).  Universal background that
   every later section leans on.  Stated, not derived.
   - **Operator vs Frobenius norm** (~5 min).  Definitions; the operator
     norm controls matvec error, the Frobenius norm controls trace-style
     variances.  Both show up later.
   - **Condition number** κ = σ_max/σ_min (~10 min).  Exercise: solve
     `Ax = b` with progressively worse κ in fp32; watch digits get lost.
     This frames why iterative methods care about spectral gaps.
   - **Machine epsilon and fp32 vs fp64** (~5 min).  Exercise: compute
     `(1+ε)−1` at the fp32 boundary.  Tees up Section 5's orthogonality
     blowup.
   - **Eigenvalues vs singular values for symmetric A; Rayleigh quotient**
     (~5 min).  Mostly prose.  Frame "power iteration = ascent on the
     Rayleigh quotient."

1. **Setting the cost currency** (~20 min).  Introduce the Hessian-vector
   product three ways: finite differences, `torch.func.jvp` of `grad`, and
   double `backward`.  Verify all three agree on a tiny MLP.  Time each.
   *Punchline:* HVP costs one forward + one backward, regardless of P.

2. **Power iteration** (~15 min).  Implement plain power iteration, plot
   convergence of the Rayleigh quotient to λ₁ on semilog axes.  Verify the
   slope matches `log(λ₂/λ₁)`.  Show what goes wrong when λ₁ ≈ −λ₂
   (sign-flipping) and when the gap is small.

3. **Deflation** (~10 min).  Subtract the rank-1 projection; recover λ₂,
   λ₃.  Note that orthogonality drift makes naïve deflation fragile past
   the first few.

4. **Lanczos: the three-term recurrence** (~25 min).  Build classical
   Lanczos from scratch.  Implement Ritz value extraction from the
   tridiagonal.  Compare Ritz values to ground-truth eigs on a small dense
   matrix where you *can* materialize and `torch.linalg.eigh`.  One
   paragraph framing **the Krylov subspace as polynomials in A**:
   `K_k(A, b) = {p(A)b : deg p < k}`, which is why Lanczos approximates
   extremes first (polynomials separate well-separated points easily).

5. **Loss of orthogonality, and how to fix it** (~30 min).  *The
   numerics-shines section.*  Run unmodified Lanczos for 50 steps on a
   moderately ill-conditioned matrix in fp32.  Plot `‖QᵀQ − Iₖ‖` and watch
   it blow up around step 20-30.  Add full reorthogonalization; then
   selective reorthogonalization (Paige).  Discuss when each is worth it.

6. **Hessian top-k in practice** (~20 min).  Apply Lanczos with selective
   reorth to the actual MLP Hessian.  Recover top-10 eigenvalues; verify
   against `torch.linalg.eigh` on the materialized Hessian (which we can
   do for the *tiny* model only, as a ground-truth check).  Discuss what
   "top" means in indefinite settings (largest |λ| vs largest λ).

### Plots (5)

- HVP timing: forward-pass cost vs HVP cost, as a function of model width.
- Power iteration convergence: log error vs iteration, slope matches theory.
- Lanczos Ritz values vs true eigenvalues at steps 5, 10, 20, 40 (the
  classic Ritz convergence picture, extremes converge first).
- **Money plot:** orthogonality loss `‖QᵀQ − I‖` over Lanczos steps, three
  curves (none / full / selective reorth) on the same axes.
- Hessian top-10 eigenvalues of the MLP at init vs after training.

---

## Notebook 2 — Randomized: sketches and the empirical NTK

**Time:** ~100 min.  Five sections, ~10 exercises.
**Anchor:** the eNTK Gram matrix of a tiny MLP on a few hundred MNIST samples.

### Sections

0. **When randomization wins: stable rank** (~10 min, 2 exercises).
   Define stable rank `r_s(A) = ‖A‖_F² / ‖A‖_2²` — the "effective rank,"
   which is small when one singular value dominates.  Exercise: compute
   `r_s` for three synthetic matrices with fast / slow / heavy-tailed
   spectral decay.  Frames the rest of the notebook: rSVD shines when
   `r_s` is small relative to ambient dimensions.

1. **Why randomize** (~15 min).  Recap Lanczos cost (matvecs are
   sequential; reorthogonalization is bookkeeping-heavy).  Pose: what if we
   trade serial matvecs for parallel ones and accept a small error?

2. **Randomized SVD on a dense toy matrix** (~25 min).  Build
   Halko-Martinsson-Tropp from scratch on a synthetic matrix with chosen
   spectral decay (fast/slow/heavy-tailed).  Sweep oversampling `p ∈
   {0, 5, 10, 20}` and plot reconstruction error vs `k`.  Add `q` power
   iterations to amplify the dominant subspace; observe how `q=1` already
   helps when decay is slow.

3. **The eNTK without materializing** (~30 min).  Derive the Novak et al.
   matvec: for `K = J Jᵀ` where `J` is the `N × P` per-sample gradient
   matrix, `Kv` is one VJP (`Σⱼ vⱼ ∇θ f(xⱼ)` gives a vector in `Rᴾ`) plus
   one JVP (push that direction through to recover `Kv ∈ Rᴺ`).  Implement
   it.  Verify against the explicit `J Jᵀ v` on a tiny case.  **Money
   plot:** time per matvec as N grows, matrix-free flat vs materialized
   exploding.  Mark the crossover point.

4. **rSVD on the eNTK** (~20 min).  Plug the matrix-free matvec into the
   range finder.  Recover top-k NTK eigenfunctions for a 500-sample MNIST
   subset.  Visualize the top eigenfunctions as images.  Compare runtime
   to "build `J`, then `np.linalg.svd`": shows the matrix-free path wins
   well before P gets large.

### Plots (4)

- rSVD error vs `k`, swept over oversampling `p`.  Three subplots for
  fast/slow/heavy spectral decay.
- Effect of power iterations `q` on slow-decay matrices.
- **Money plot:** matrix-free vs materialized eNTK matvec time vs N.
- Top-6 eNTK eigenfunctions as MNIST-shaped images.

---

## Notebook 3 — Estimation: trace, density of states, perturbation

**Time:** ~115 min.  Five sections, ~11 exercises.
**Anchor:** the Hessian of the tiny MLP from Notebook 1, plus the eNTK
from Notebook 2 for the closing perturbation experiment.

### Sections

1. **Hutchinson's trick** (~20 min).  Derive `tr(A) = E[zᵀAz]` for `E[zzᵀ]
   = I`.  Implement with Rademacher and Gaussian probes.  Plot empirical
   variance vs `m`; verify the rates:
   - Gaussian:    `Var = 2‖A‖_F² / m`        (any symmetric A)
   - Rademacher:  `Var = 2(‖A‖_F² − ‖diag(A)‖²) / m`  (lower whenever A has
     diagonal mass)
   Mention Hutch++ as a one-paragraph pointer; do not implement.

2. **The Hessian's trace and its meaning** (~15 min).  `tr(H)` is the sum
   of eigenvalues; for the MLP this is dominated by the bulk, not the
   outliers.  Compute it with Hutchinson; verify against `torch.diag`-and-
   sum on the materialized Hessian for the tiny model.

3. **From point estimates to the spectral density** (~30 min).  Motivate:
   we want the *histogram* of eigenvalues, not just statistics.  Build
   stochastic Lanczos quadrature: for each probe `z`, run `s`-step Lanczos
   starting at `z`, take the Ritz values and Lanczos weights, accumulate
   into a Gaussian-smoothed density.  Implement; verify against
   ground-truth eigenvalue histogram on the tiny model.

4. **DOS in practice** (~25 min).  Apply SLQ to the MLP Hessian at init
   and after training.  **Money plot:** reproduce the Ghorbani-style
   bulk-near-zero-plus-outliers picture, on a log-y axis to see both
   regimes.  Discuss: outliers are roughly the Lanczos top-k from
   Notebook 1; the bulk is what Lanczos *cannot* see efficiently.

5. **Eigenvalue perturbation: how much do you trust this?** (~30 min, 3
   exercises).  Closes the notebook by framing every estimate above as a
   perturbed version of an idealized object.
   - **Weyl's inequality** (~8 min).  For symmetric A and perturbation E,
     `|λ_k(A+E) − λ_k(A)| ≤ ‖E‖_2`.  Exercise: pick a 50×50 symmetric
     matrix, perturb by `αE` over a grid of α, plot `|Δλ_k|` against
     `α·‖E‖`.  Confirm the bound, and that it's tight in the worst case.
     Connect back: Notebook 1's "Lanczos in fp32 with selective reorth"
     has effective perturbation `≈ √ε_mach · ‖A‖` — Weyl tells you the
     eigenvalues are protected to that scale.
   - **Davis-Kahan sin Θ** (~12 min).  Principal angle between the
     true and perturbed top-k eigenspaces is bounded by `‖E‖ / gap`,
     where the gap is the separation between the targeted spectrum and
     the rest.  Exercise: construct A with adjustable eigenvalue gap,
     perturb, measure principal angle, watch it diverge as the gap
     shrinks.  This is the "near-degenerate subspaces rotate freely"
     phenomenon.
   - **Bootstrap on the eNTK** (~10 min).  Resample 80% of the MNIST
     subset 20 times, recompute top-5 eNTK eigenvalues and eigenvectors
     each time.  Plot CI for eigenvalues; plot pairwise principal angle
     for the corresponding eigenvectors.  **Punchline:** the eigenvalues
     are tight; the eigenvectors are not, exactly when gaps are small.
     That's the ML reader's takeaway — trust eigenvalue summaries, be
     cautious with eigenvector summaries.

### Plots (6)

- Hutchinson variance vs `m`, theoretical line overlaid, two probe types.
- Bias-variance tradeoff for SLQ smoothing width.
- **Money plot:** DOS at init vs after training, log-y, bulk + outliers
  clearly visible.
- (Auxiliary) Lanczos Ritz values overlaid on the SLQ density at trained
  state — shows what Krylov sees and what it misses.
- Weyl bound: `|Δλ_k|` vs `α·‖E‖`, with the `y = x` ceiling.
- Davis-Kahan: principal angle vs `‖E‖ / gap`, plus bootstrap CIs
  for eNTK eigenvalues alongside pairwise angles for eigenvectors
  (same figure, two panels — the "eigenvalues stable, eigenvectors not"
  story in one image).

---

## Notebook 4 — Capstone: spectroscopy of a small CNN

**Time:** ~120 min.  Looser structure, more open-ended exercises.
**Anchor:** a small CNN (~30k params) trained on a 1000-sample CIFAR-10
subset.  Pretrained checkpoint shipped; training script provided but the
notebook reads checkpoints rather than retraining.

### Sections

1. **Setup** (~15 min).  Load the checkpoints at epochs `{0, 1, 10, 100}`.
   Sanity-check accuracy and loss.

2. **Hessian top-k via Lanczos** (~25 min).  Compute top-10 eigenvalues at
   each checkpoint.  Plot evolution across training.  Are outliers growing,
   shrinking, splitting?

3. **Hessian trace and DOS via Hutchinson + SLQ** (~25 min).  Same
   checkpoints.  Plot DOS at each, stacked.  Watch the bulk consolidate
   and the outliers separate.

4. **eNTK top modes via rSVD** (~25 min).  Compute the top-6 eNTK
   eigenfunctions at each checkpoint.  Visualize as CIFAR-shaped images.
   Are they becoming more "feature-like" through training?

5. **Synthesis** (~20 min).  Two open-ended prompts:
   - **Cost ledger:** for each of the four computations above, fill in
     concrete matvec counts and wall-clock numbers.  Where would the
     budget bind if this model were 1000× larger?
   - **What's missing:** name three things you can *not* read off these
     spectral summaries.  (Suggested answers in the solutions file:
     eigenvector localization, anisotropy across data points, off-diagonal
     coupling between layers.)

6. **Pointers** (~10 min).  One paragraph each on: influence functions
   (need Hessian-inverse-vector products, hence CG → followup tutorial);
   K-FAC and friends (block-diagonal approximations of the inverse Fisher);
   the lazy / NTK regime (when does the eNTK *predict* training?).

### Plots (~6)

- Hessian top-10 eigenvalues across training checkpoints (line plot, k
  curves).
- DOS across training (small multiples, log-y).
- Trace across training, alongside `‖∇L‖²` for comparison.
- Top-6 eNTK eigenfunctions at each checkpoint (grid of images).
- Hessian-eNTK comparison: top eigvecs of one projected onto the other.
- A "cost ledger" bar chart: matvec count per technique, on the actual
  model.

---

## Shared infrastructure (`src/`)

- `tiny_models.py`: three deterministic models, in increasing size.
  - `toy_mlp()`: 20-dim input, width 8, depth 2, ~250 params.  Used for
    ground-truth eigendecompositions (materializing a 250×250 Hessian is
    cheap).
  - `tiny_mlp()`: input 7×7 downsampled MNIST → width 32, depth 2, ~2k
    params.  Used as the "realistic but Hessian-still-materializable"
    workhorse for Notebooks 1-3.  A 2k×2k Hessian is ~16 MB.
  - `tiny_cnn()`: ~30k params, CIFAR-10 32×32×3 input.  Hessian is *not*
    materialized for this one — that's the whole point of the capstone.
- `hvp.py`: a reference HVP and a reference eNTK matvec.  Notebooks 1-3
  ask the reader to *write their own* and then `assert torch.allclose(...)`
  against these references.
- `plotting.py`: a single matplotlib style.  Helper for the "money plots"
  so all four notebooks render consistently.
- `tests.py`: small assert wrappers used inline in the notebooks
  (`tests.test_hvp(your_hvp)`).
- `data/`: cached MNIST-1k and CIFAR-1k subsets as `.pt` files so the
  notebooks don't hit torchvision on every run.

## Length and effort target

- Total writing: ~6000-8000 words of prose across the four notebooks plus
  ~1500 words in the README.
- ~30-40 exercises total, sized 3-15 min each.
- All cells must run in under 30 s on a laptop CPU.  The capstone is the
  exception: its top-k Lanczos may take ~1-2 minutes per checkpoint.
- The pretrained capstone checkpoints (4 files, ~30k params each) ship in
  the repo; small enough to commit (<1 MB total).

## Open questions / things to revisit during implementation

- Should the "money plots" be regenerated by the notebooks themselves, or
  baked-in PNGs?  Lean toward "regenerated, with a cached fallback if the
  computation budget gets tight."
- Capstone uses CIFAR-10; smaller would be honest (MNIST) but CIFAR makes
  the eNTK eigenfunctions visually richer.  Default CIFAR-10 1k subset.
- Whether to ship a `uv`-managed environment or just a `requirements.txt`.
  Lean `uv` (matches other tutorials in `~/Programming/Claude/tutorials/`).
