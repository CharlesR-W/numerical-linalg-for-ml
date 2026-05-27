"""Train tiny_mlp on 7x7 MNIST, save checkpoints at strategic steps.

Used by Notebook 4 (the capstone).  Checkpoints are committed to the repo
so the notebook can load them without retraining.

Run: `uv run python scripts/train_capstone.py`
Output: data/checkpoints/step_{0,10,100,500}.pt
"""
from pathlib import Path

import torch
import torch.nn.functional as F

from src.data import load_mnist_7x7
from src.tiny_models import tiny_mlp

CHECKPOINTS_AT = [0, 10, 100, 500]
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "checkpoints"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    model = tiny_mlp(seed=0)
    X, y = load_mnist_7x7(n=500, seed=0)
    opt = torch.optim.SGD(model.parameters(), lr=0.1)

    last_step = max(CHECKPOINTS_AT)
    for step in range(last_step + 1):
        if step in CHECKPOINTS_AT:
            loss_val = F.cross_entropy(model(X), y).item()
            ckpt_path = OUT_DIR / f"step_{step}.pt"
            torch.save({
                "state_dict": {k: v.clone() for k, v in model.state_dict().items()},
                "step": step,
                "loss": loss_val,
            }, ckpt_path)
            print(f"step {step:5d}: loss = {loss_val:.4f}  →  {ckpt_path.name}")
        if step == last_step:
            break
        idx = torch.randint(0, len(X), (64,))
        opt.zero_grad()
        F.cross_entropy(model(X[idx]), y[idx]).backward()
        opt.step()

    print(f"\nwrote {len(CHECKPOINTS_AT)} checkpoints to {OUT_DIR}")


if __name__ == "__main__":
    main()
