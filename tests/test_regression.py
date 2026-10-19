"""Regression: a frozen reference image must keep the same label/confidence per model.

Fixtures (see the one-time setup commands):
  tests/fixtures/reference.png           112x112 upscaled Fashion-MNIST test image
  tests/fixtures/reference_expected.json {"baseline": {...}, "improved": {...}}
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from api.gradcam import predict_with_cam

FIX = Path(__file__).parent / "fixtures"
REF_IMG, REF_JSON = FIX / "reference.png", FIX / "reference_expected.json"

pytestmark = pytest.mark.skipif(
    not (REF_IMG.exists() and REF_JSON.exists()),
    reason="reference fixtures missing (see one-time setup commands)",
)


def _ref28() -> Image.Image:
    # The reference was upscaled 4x with NEAREST, so NEAREST back down is lossless.
    return Image.open(REF_IMG).convert("L").resize((28, 28), Image.NEAREST)


@pytest.mark.parametrize("model", ["baseline", "improved"])
def test_reference_prediction_is_stable(model):
    expected = json.loads(REF_JSON.read_text())[model]
    out = predict_with_cam(model, np.asarray(_ref28(), dtype=np.uint8))
    assert out["label"] == expected["label"]
    assert out["confidence"] == pytest.approx(expected["confidence"], abs=1e-3)


@pytest.mark.parametrize("model", ["baseline", "improved"])
def test_reference_prediction_is_stable_through_api(client, model):
    expected = json.loads(REF_JSON.read_text())[model]
    buf = io.BytesIO()
    _ref28().save(buf, format="PNG")
    resp = client.post("/predict", files={"file": ("ref.png", buf.getvalue(), "image/png")},
                       data={"model": model})
    body = resp.json()
    assert resp.status_code == 200
    assert body["label"] == expected["label"]
    assert body["confidence"] == pytest.approx(expected["confidence"], abs=1e-3)