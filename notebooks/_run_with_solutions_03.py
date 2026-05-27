"""Inject solutions into 03_estimation.ipynb and execute it."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).parent / "03_estimation.ipynb"

INJECT = """

# === INJECTED ===
from solutions._03_estimation import (
    hutchinson_trace as _ref_hutchinson,
    slq_density as _ref_slq,
)
hutchinson_trace = _ref_hutchinson
slq_density = _ref_slq
# === END ===

"""


def main():
    nb = nbformat.read(NB, as_version=4)
    for cell in nb.cells:
        if cell.cell_type != "code":
            continue
        if "raise NotImplementedError" in cell.source:
            cell.source = cell.source.replace(
                "    raise NotImplementedError\n\n",
                "    raise NotImplementedError\n" + INJECT,
            )
            if cell.source.endswith("    raise NotImplementedError"):
                cell.source = cell.source + "\n" + INJECT

    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    print(f"✓ notebook 3 executed end-to-end ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
