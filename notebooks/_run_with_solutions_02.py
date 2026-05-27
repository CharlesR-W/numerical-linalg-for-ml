"""Inject solutions into 02_randomized.ipynb and execute it."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).parent / "02_randomized.ipynb"

INJECT = """

# === INJECTED ===
from solutions._02_randomized import (
    stable_rank as _ref_stable_rank,
    randomized_eigh as _ref_rsvd,
    entk_matvec as _ref_entk_matvec,
)
stable_rank = _ref_stable_rank
randomized_eigh = _ref_rsvd
entk_matvec = _ref_entk_matvec
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

    client = NotebookClient(nb, timeout=300, kernel_name="python3")
    client.execute()
    print(f"✓ notebook 2 executed end-to-end ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
