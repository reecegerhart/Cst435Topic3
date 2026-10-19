"""POST /predict: multipart smoke test, Grad-CAM overlay shape, privacy, logging."""
from __future__ import annotations

import base64
import hashlib
import io

import pytest
from PIL import Image

from shared.data import N_CLASSES
from tests.conftest import make_image_bytes

MIME = {"PNG": "image/png", "JPEG": "image/jpeg"}


def _predict(client, raw, fmt="PNG", model=None):
    kwargs = {"files": {"file": (f"up.{fmt.lower()}", raw, MIME[fmt])}}
    if model:
        kwargs["data"] = {"model": model}
    return client.post("/predict", **kwargs)


def _decode(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_predict_multipart_smoke(client):
    raw = make_image_bytes(64, 64)
    resp = _predict(client, raw)
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_name"] == "improved"          # default model
    assert body["label"] in range(N_CLASSES)
    assert len(body["probs"]) == N_CLASSES
    assert sum(body["probs"]) == pytest.approx(1.0, abs=1e-4)
    assert len(body["top5"]) == 5
    assert body["top5"][0]["label"] == body["label"]
    assert body["image_sha256"] == hashlib.sha256(raw).hexdigest()
    assert body["logged"] is True and body["run_id"] == 2


@pytest.mark.parametrize("model,run_id", [("baseline", 1), ("improved", 2)])
def test_model_selection_links_the_right_run(client, model, run_id):
    body = _predict(client, make_image_bytes(), model=model).json()
    assert body["model_name"] == model
    assert body["run_id"] == run_id


@pytest.mark.parametrize("w,h,mode,fmt", [
    (28, 28, "L", "PNG"),
    (217, 360, "RGB", "PNG"),
    (100, 50, "RGBA", "PNG"),
    (512, 300, "RGB", "PNG"),
    (123, 77, "RGB", "JPEG"),
])
def test_gradcam_overlay_matches_input_dimensions(client, w, h, mode, fmt):
    resp = _predict(client, make_image_bytes(w, h, mode, fmt), fmt=fmt)
    assert resp.status_code == 200
    body = resp.json()
    assert (body["width"], body["height"]) == (w, h)
    assert _decode(body["cam_png"]).size == (w, h)
    assert _decode(body["input_png"]).size == (w, h)


def test_only_hash_and_metadata_are_logged(client):
    raw = make_image_bytes(40, 30)
    body = _predict(client, raw).json()
    pred, meta = client._store["predictions"][-1], client._store["image_metadata"][-1]
    assert pred["image_hash"] == meta["image_hash"] == hashlib.sha256(raw).hexdigest()
    assert (meta["width"], meta["height"], meta["channels"]) == (40, 30, 3)
    assert meta["prediction_id"] == body["prediction_id"]
    for row in (pred, meta):                          # no pixel payloads anywhere
        assert not any("png" in k or "pixel" in k or "bytes" in k for k in row)


def test_prediction_still_returned_if_logging_fails(client, monkeypatch):
    from api import db

    def boom(**_kw):
        raise RuntimeError("supabase down")

    monkeypatch.setattr(db, "insert_prediction", boom)
    resp = _predict(client, make_image_bytes())
    assert resp.status_code == 200
    body = resp.json()
    assert body["logged"] is False and body["prediction_id"] is None