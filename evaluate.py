"""Evaluate both saved models on the Fashion-MNIST TEST split.

    python evaluate.py            # plots, metrics, gallery
    python evaluate.py --push     # also store gallery info in Supabase runs.gallery

Outputs:
  report_assets/<name>_curves.png      train/val loss + accuracy curves
  report_assets/<name>_confusion.png   confusion matrix (test)
  report_assets/metrics.json           test acc, confusion matrices, per-class P/R
  report_assets/per_class.md           per-class precision/recall table (paste into README)
  gallery/test_<idx>.png, gallery/gallery.json   Compare Models gallery
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from api.models import build_model
from shared.data import CLASS_NAMES, N_CLASSES, load_fashion_mnist
from train import to_tensor

WEIGHTS, OUT, GALLERY = Path("weights"), Path("report_assets"), Path("gallery")
MODELS = ["baseline", "improved"]


@torch.no_grad()
def predict_probs(model, x_uint8: np.ndarray, batch: int = 1000) -> np.ndarray:
    xs = to_tensor(x_uint8)
    out = [torch.softmax(model(xs[i : i + batch]), dim=1) for i in range(0, len(xs), batch)]
    return torch.cat(out).numpy()


def confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    np.add.at(cm, (y_true, y_pred), 1)  # rows = true, cols = predicted
    return cm


def precision_recall(cm: np.ndarray):
    tp = np.diag(cm).astype(float)
    return tp / np.maximum(cm.sum(0), 1), tp / np.maximum(cm.sum(1), 1)


def plot_curves(name: str, hist: dict) -> None:
    ep = range(1, len(hist["train_loss"]) + 1)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, key, title in [(ax[0], "loss", "Loss"), (ax[1], "acc", "Accuracy")]:
        a.plot(ep, hist[f"train_{key}"], marker="o", label="train")
        a.plot(ep, hist[f"val_{key}"], marker="o", label="validation")
        a.set_xlabel("Epoch"); a.set_title(f"{name}: {title}"); a.legend(); a.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(OUT / f"{name}_curves.png", dpi=150); plt.close(fig)


def plot_confusion(name: str, cm: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{name}: confusion matrix (test)")
    fig.tight_layout(); fig.savefig(OUT / f"{name}_confusion.png", dpi=150); plt.close(fig)


def build_gallery(x_test, y_test, probs, rng, n_same=5, n_diff=3):
    pred = {k: p.argmax(1) for k, p in probs.items()}
    b_ok, i_ok = pred["baseline"] == y_test, pred["improved"] == y_test
    groups = {
        "both_correct": (np.where(b_ok & i_ok)[0], n_same),
        "both_wrong": (np.where(~b_ok & ~i_ok)[0], n_same),
        "baseline_only_correct": (np.where(b_ok & ~i_ok)[0], n_diff),
        "improved_only_correct": (np.where(~b_ok & i_ok)[0], n_diff),
    }
    GALLERY.mkdir(exist_ok=True)
    items = []
    for cat, (idxs, n) in groups.items():
        for idx in sorted(rng.choice(idxs, size=min(n, len(idxs)), replace=False)):
            idx = int(idx)
            Image.fromarray(x_test[idx]).resize((112, 112), Image.NEAREST).save(GALLERY / f"test_{idx}.png")
            item = {"test_index": idx, "category": cat,
                    "true_label": int(y_test[idx]), "true_class": CLASS_NAMES[int(y_test[idx])]}
            for k in MODELS:
                lab = int(pred[k][idx])
                item[k] = {"label": lab, "class": CLASS_NAMES[lab], "confidence": float(probs[k][idx, lab])}
            items.append(item)
    (GALLERY / "gallery.json").write_text(json.dumps(items, indent=2))
    return items


def push_gallery(items) -> None:
    from dotenv import load_dotenv
    from supabase import create_client

    load_dotenv()
    sb = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    for name in MODELS:
        slim = [{"test_index": it["test_index"], "category": it["category"],
                 "true_label": it["true_label"], "pred_label": it[name]["label"],
                 "confidence": it[name]["confidence"]} for it in items]
        res = sb.table("runs").update({"gallery": slim}).eq("name", name).execute()
        print(f"{'OK  ' if res.data else 'MISS'} gallery -> runs.name={name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    _, _, x_test, y_test = load_fashion_mnist(args.data_dir)

    probs, metrics = {}, {}
    for name in MODELS:
        model = build_model(name)
        model.load_state_dict(torch.load(WEIGHTS / f"{name}.pt", map_location="cpu"))
        model.eval()
        probs[name] = predict_probs(model, x_test)
        pred = probs[name].argmax(1)
        cm = confusion(y_test, pred)
        p, r = precision_recall(cm)
        metrics[name] = {
            "test_acc": float((pred == y_test).mean()),
            "confusion_matrix": cm.tolist(),
            "per_class": {CLASS_NAMES[c]: {"precision": float(p[c]), "recall": float(r[c])}
                          for c in range(N_CLASSES)},
        }
        hist = json.loads((WEIGHTS / f"{name}_run.json").read_text())["history"]
        plot_curves(name, hist)
        plot_confusion(name, cm)
        print(f"{name}: test acc {metrics[name]['test_acc']:.4f}")

    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    lines = ["| Class | Baseline P | Baseline R | Improved P | Improved R |", "|---|---|---|---|---|"]
    for c in CLASS_NAMES:
        b, i = metrics["baseline"]["per_class"][c], metrics["improved"]["per_class"][c]
        lines.append(f"| {c} | {b['precision']:.3f} | {b['recall']:.3f} | {i['precision']:.3f} | {i['recall']:.3f} |")
    (OUT / "per_class.md").write_text("\n".join(lines) + "\n")

    items = build_gallery(x_test, y_test, probs, np.random.default_rng(args.seed))
    print(f"Gallery: {len(items)} images in {GALLERY}/")
    if args.push:
        push_gallery(items)


if __name__ == "__main__":
    main()