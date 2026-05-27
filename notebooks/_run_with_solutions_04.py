"""Capstone smoke test - no NotImplementedError stubs to inject (all uses
of reference functions are explicit imports from solutions/)."""
from pathlib import Path

import nbformat
from nbclient import NotebookClient

NB = Path(__file__).parent / "04_capstone.ipynb"


def main():
    nb = nbformat.read(NB, as_version=4)
    client = NotebookClient(nb, timeout=900, kernel_name="python3")
    client.execute()
    print(f"✓ notebook 4 (capstone) executed end-to-end ({len(nb.cells)} cells)")


if __name__ == "__main__":
    main()
