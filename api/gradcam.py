"""Model loading, prediction, and Grad-CAM for the API (Render).

Loads plain state_dict weights from weights/ (no pickle). Grad-CAM hooks the
model's `cam_layer` (last conv block's activations, 14x14), so the heatmap is
finer than the final 7x7 feature map.

    Run a quick visual check locally:   python -m api.gradcam
"""
from __future__ import annotations

import base64
import io
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from PIL import Image

from api.models import build_model
from shared.data import CLASS_NAMES, IMG_SIZE

WEIGHTS_DIR = Path(os.environ.get("WEIGHTS_DIR", Path(__file__).resolve().parent.parent / "weights"))


@lru_cache(maxsize=None)
def get_model(name: str):
    """Load (and cache) a trained model by name: 'baseline' or 'improved'."""
    model = build_model(name)
    state = torch.load(WEIGHTS_DIR / f"{name}.pt", map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _jet(v: np.ndarray) -> np.ndarray:
    """Tiny jet-style colormap, [0,1] -> [H, W, 3] float in [0,255] (no matplotlib needed)."""
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return np.stack([r, g, b], axis=-1) * 255.0


def _png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay(gray_uint8: np.ndarray, cam01: np.ndarray, out_size: Tuple[int, int]) -> Image.Image:
    """Blend a heatmap over the grayscale input at out_size = (width, height)."""
    w, h = out_size
    cam = Image.fromarray((cam01 * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    heat = _jet(np.asarray(cam, dtype=np.float32) / 255.0)
    base = np.asarray(Image.fromarray(gray_uint8, "L").resize((w, h), Image.BILINEAR).convert("RGB"),
                      dtype=np.float32)
    return Image.fromarray(np.clip(0.55 * base + 0.45 * heat, 0, 255).astype(np.uint8), "RGB")


def predict_with_cam(
    model_name: str,
    gray_uint8: np.ndarray,
    out_size: Optional[Tuple[int, int]] = None,
) -> Dict[str, object]:
    """Classify one [28, 28] uint8 image and build the Grad-CAM overlay.

    ``out_size`` is (width, height) for the returned PNGs; defaults to 28x28.
    The returned dict has no run_id / image hash; main.py adds those.
    """
    assert gray_uint8.shape == (IMG_SIZE, IMG_SIZE) and gray_uint8.dtype == np.uint8
    model = get_model(model_name)
    out_size = out_size or (IMG_SIZE, IMG_SIZE)
    x = torch.from_numpy(gray_uint8.astype(np.float32) / 255.0)[None, None]  # [1,1,28,28]

    store: Dict[str, torch.Tensor] = {}

    def fwd_hook(_module, _inp, out):
        store["act"] = out
        out.register_hook(lambda g: store.__setitem__("grad", g))

    handle = model.cam_layer.register_forward_hook(fwd_hook)
    try:
        model.zero_grad()
        logits = model(x)                        # one forward pass serves both outputs
        probs_t = torch.softmax(logits, dim=1)[0].detach()
        label = int(probs_t.argmax())
        logits[0, label].backward()
    finally:
        handle.remove()

    act = store["act"][0].detach()               # [C, h, w]
    weights = store["grad"][0].mean(dim=(1, 2))  # [C] global-average-pooled gradients
    cam = torch.relu((weights[:, None, None] * act).sum(0))
    cam = (cam / (cam.max() + 1e-8)).numpy()

    probs = [float(p) for p in probs_t.tolist()]
    top5 = [{"label": int(i), "class_name": CLASS_NAMES[int(i)], "prob": probs[int(i)]}
            for i in np.argsort(probs)[::-1][:5]]
    input_img = Image.fromarray(gray_uint8, "L").resize(out_size, Image.BILINEAR)
    return {
        "model_name": model_name,
        "label": label,
        "class_name": CLASS_NAMES[label],
        "confidence": probs[label],
        "probs": probs,
        "top5": top5,
        "input_png": _png_b64(input_img),
        "cam_png": _png_b64(_overlay(gray_uint8, cam, out_size)),
    }


if __name__ == "__main__":
    # Visual sanity check: Grad-CAM for the first 4 test images, both models.
    from shared.data import load_fashion_mnist

    _, _, x_test, y_test = load_fashion_mnist("data")
    out = Path("report_assets"); out.mkdir(exist_ok=True)
    for name in ("baseline", "improved"):
        tiles = []
        for i in range(4):
            r = predict_with_cam(name, x_test[i], out_size=(168, 168))
            print(f"{name} test[{i}]: true={CLASS_NAMES[y_test[i]]:12s} pred={r['class_name']:12s} conf={r['confidence']:.2f}")
            tiles.append(np.asarray(Image.open(io.BytesIO(base64.b64decode(r["cam_png"])))))
        Image.fromarray(np.concatenate(tiles, axis=1)).save(out / f"gradcam_demo_{name}.png")
    print("Saved report_assets/gradcam_demo_*.png")