"""Train the baseline or improved CNN on Fashion-MNIST. Run this on Colab.

    python train.py --model baseline
    python train.py --model improved

Both models get the same budget: epochs, batch size, base LR, and seed.
Outputs: weights/<name>.pt, weights/<name>_run.json, and a row in Supabase `runs`
if SUPABASE_URL and SUPABASE_SERVICE_KEY are set in the environment.

Note: train_acc/train_loss are running averages over the epoch in train mode
(with augmentation and dropout active), so they are not directly comparable to
val metrics, which are computed in eval mode.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from api.models import build_model
from shared.data import IMG_SIZE, load_fashion_mnist, make_splits

WEIGHTS_DIR = Path("weights")

# Per-model recipe. Baseline = none of the tricks; improved adds weight decay,
# a cosine LR schedule, and augmentation (BN + dropout are in the architecture).
PRESETS = {
    "baseline": {"weight_decay": 0.0, "scheduler": "none", "augment": False},
    "improved": {"weight_decay": 1e-2, "scheduler": "cosine", "augment": True},
}


def to_tensor(x: np.ndarray) -> torch.Tensor:
    """[N, H, W] uint8 -> [N, 1, H, W] float32 in [0, 1]."""
    return torch.from_numpy(x.astype(np.float32) / 255.0).unsqueeze(1)


def augment(x: torch.Tensor, max_shift_px: int = 2) -> torch.Tensor:
    """Batched random horizontal flip + random shift of up to +/-2 px."""
    n, dev = x.size(0), x.device
    flip = (torch.rand(n, device=dev) < 0.5).float() * 2 - 1      # -1 = flip
    shift = (torch.rand(n, 2, device=dev) * 2 - 1) * (max_shift_px * 2 / IMG_SIZE)
    theta = torch.zeros(n, 2, 3, device=dev)
    theta[:, 0, 0] = flip
    theta[:, 1, 1] = 1.0
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


@torch.no_grad()
def evaluate(model, x, y, device, batch_size: int = 1000):
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(x), batch_size):
        xb, yb = x[i : i + batch_size].to(device), y[i : i + batch_size].to(device)
        logits = model(xb)
        total_loss += F.cross_entropy(logits, yb, reduction="sum").item()
        correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(x), correct / len(x)


def log_run_to_supabase(row: dict):
    url, key = os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_SERVICE_KEY")
    if not (url and key):
        print("Supabase env vars not set; skipping DB insert (run JSON saved locally).")
        return
    from supabase import create_client

    res = create_client(url, key).table("runs").insert(row).execute()
    print(f"Logged to Supabase: runs.id = {res.data[0]['id']}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["baseline", "improved"], required=True)
    p.add_argument("--name", default=None, help="defaults to the model kind")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    name = args.name or args.model
    preset = PRESETS[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    print(f"Training '{name}' on {device}: {args.epochs} epochs, bs={args.batch_size}, lr={args.lr}")

    x_tr, y_tr, x_te, y_te = load_fashion_mnist(args.data_dir)
    splits = make_splits(x_tr, y_tr, x_te, y_te, seed=args.seed)
    data = {k: (to_tensor(x), torch.from_numpy(y)) for k, (x, y) in splits.items()}
    x_train, y_train = data["train"][0].to(device), data["train"][1].to(device)

    model = build_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=preset["weight_decay"])
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        if preset["scheduler"] == "cosine" else None
    )

    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "lr": []}
    n = x_train.size(0)
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        run_loss, run_correct = 0.0, 0
        for s in range(0, n, args.batch_size):
            idx = perm[s : s + args.batch_size]
            xb, yb = x_train[idx], y_train[idx]
            if preset["augment"]:
                xb = augment(xb)
            opt.zero_grad()
            logits = model(xb)
            loss = F.cross_entropy(logits, yb)
            loss.backward()
            opt.step()
            run_loss += loss.item() * xb.size(0)
            run_correct += (logits.argmax(1) == yb).sum().item()
        hist["lr"].append(opt.param_groups[0]["lr"])
        if sched:
            sched.step()
        val_loss, val_acc = evaluate(model, *data["val"], device)
        hist["train_loss"].append(run_loss / n)
        hist["train_acc"].append(run_correct / n)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        print(f"epoch {epoch:2d}/{args.epochs}  train_loss {hist['train_loss'][-1]:.4f}  "
              f"train_acc {hist['train_acc'][-1]:.4f}  val_loss {val_loss:.4f}  val_acc {val_acc:.4f}")

    test_loss, test_acc = evaluate(model, *data["test"], device)
    print(f"TEST  loss {test_loss:.4f}  acc {test_acc:.4f}  ({time.time() - t0:.0f}s total)")

    WEIGHTS_DIR.mkdir(exist_ok=True)
    weights_path = WEIGHTS_DIR / f"{name}.pt"
    torch.save(model.cpu().state_dict(), weights_path)

    row = {
        "name": name,
        "model_type": args.model,
        "dataset": "fashion_mnist",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": preset["weight_decay"],
        "seed": args.seed,
        "hyperparams": {
            "optimizer": "AdamW",
            "scheduler": preset["scheduler"],
            "augmentation": preset["augment"],
            "batch_norm": args.model == "improved",
            "dropout": 0.3 if args.model == "improved" else 0.0,
        },
        "train_loss": hist["train_loss"][-1],
        "train_acc": hist["train_acc"][-1],
        "val_loss": hist["val_loss"][-1],
        "val_acc": hist["val_acc"][-1],
        "test_loss": test_loss,
        "test_acc": test_acc,
        "history": hist,
        "weights_uri": weights_path.as_posix(),
    }
    (WEIGHTS_DIR / f"{name}_run.json").write_text(json.dumps(row, indent=2))
    log_run_to_supabase(row)


if __name__ == "__main__":
    main()