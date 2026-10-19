"""Request validation and read-endpoint tests."""
from __future__ import annotations

from tests.conftest import make_image_bytes


def _post(client, raw, **data):
    return client.post("/predict", files={"file": ("x.png", raw, "image/png")}, data=data)


def test_predict_requires_a_file(client):
    assert client.post("/predict", data={"model": "improved"}).status_code == 422


def test_predict_rejects_empty_file(client):
    assert _post(client, b"").status_code == 422


def test_predict_rejects_non_image(client):
    assert _post(client, b"this is not an image").status_code == 422


def test_predict_rejects_unknown_model(client):
    assert _post(client, make_image_bytes(), model="resnet").status_code == 422


def test_predict_rejects_oversized_dimensions(client):
    assert _post(client, make_image_bytes(1025, 10)).status_code == 422


def test_predict_rejects_oversized_file(client):
    assert _post(client, b"\0" * (5 * 1024 * 1024 + 1)).status_code == 413


def test_runs_listing_and_lookup(client):
    runs = client.get("/runs").json()
    assert {r["model_type"] for r in runs} == {"baseline", "improved"}
    assert client.get("/runs/2").json()["name"] == "improved"
    assert client.get("/runs/999").status_code == 404


def test_limit_bounds_are_validated(client):
    assert client.get("/runs", params={"limit": 0}).status_code == 422
    assert client.get("/predictions", params={"limit": 101}).status_code == 422


def test_predictions_are_joined_to_runs(client):
    r = _post(client, make_image_bytes(), model="baseline")
    assert r.status_code == 200
    rows = client.get("/predictions").json()
    assert rows[0]["id"] == r.json()["prediction_id"]
    assert rows[0]["runs"]["id"] == 1 and rows[0]["runs"]["model_type"] == "baseline"