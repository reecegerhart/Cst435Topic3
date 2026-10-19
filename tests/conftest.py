"""Shared pytest fixtures.

The `client` fixture runs the REAL FastAPI app and the REAL models, but swaps the
Supabase layer for an in-memory fake, so the suite needs no network. The live
round-trip test (test_supabase_roundtrip.py) skips itself without credentials.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pytest
from dotenv import load_dotenv
from PIL import Image

load_dotenv()  # so has_supabase_creds() sees a local .env


def has_supabase_creds() -> bool:
    return bool(os.environ.get("SUPABASE_URL") and os.environ.get("SUPABASE_SERVICE_KEY"))


def make_image_bytes(width: int = 64, height: int = 64, mode: str = "RGB",
                     fmt: str = "PNG", seed: int = 0) -> bytes:
    """Deterministic random-noise image encoded as PNG/JPEG bytes."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr, "RGB").convert(mode)
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def _fake_run(run_id: int, name: str, model_type: str, test_acc: float) -> dict:
    return {
        "id": run_id, "name": name, "model_type": model_type, "dataset": "fashion_mnist",
        "epochs": 10, "batch_size": 64, "lr": 1e-3, "weight_decay": 0.0,
        "weights_uri": f"weights/{name}.pt", "test_acc": test_acc,
        "created_at": "2026-01-01T00:00:00+00:00",
    }


@pytest.fixture
def client(monkeypatch):
    """TestClient with Supabase replaced by an in-memory store (exposed as client._store)."""
    from fastapi.testclient import TestClient

    from api import db

    store = {
        "runs": {1: _fake_run(1, "baseline", "baseline", 0.9099),
                 2: _fake_run(2, "improved", "improved", 0.8933)},
        "predictions": [],
        "image_metadata": [],
    }

    def get_run(run_id):
        return store["runs"].get(run_id)

    def get_run_for_model(model_type):
        rows = [r for r in store["runs"].values() if r["model_type"] == model_type]
        return rows[-1] if rows else None

    def latest_runs(limit=20):
        return list(store["runs"].values())[::-1][:limit]

    def insert_prediction(*, image_hash, width, height, channels, label, class_name, top5, run_id):
        row = {
            "id": len(store["predictions"]) + 1, "image_hash": image_hash,
            "predicted_label": label, "predicted_class": class_name, "top5": top5,
            "served_by_run_id": run_id, "created_at": "2026-01-01T00:00:00+00:00",
        }
        store["predictions"].append(row)
        store["image_metadata"].append({
            "id": len(store["image_metadata"]) + 1, "prediction_id": row["id"],
            "image_hash": image_hash, "width": width, "height": height,
            "channels": channels, "label": class_name,
        })
        return row

    def latest_predictions(limit=50):
        out = []
        for p in store["predictions"][::-1][:limit]:
            run = store["runs"][p["served_by_run_id"]]
            out.append({**p, "runs": {k: run[k] for k in ("id", "name", "model_type", "test_acc")}})
        return out

    monkeypatch.setattr(db, "ping", lambda: True)
    monkeypatch.setattr(db, "get_run", get_run)
    monkeypatch.setattr(db, "get_run_for_model", get_run_for_model)
    monkeypatch.setattr(db, "latest_runs", latest_runs)
    monkeypatch.setattr(db, "insert_prediction", insert_prediction)
    monkeypatch.setattr(db, "latest_predictions", latest_predictions)

    from api.main import app

    with TestClient(app) as c:
        c._store = store
        yield c