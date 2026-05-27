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
