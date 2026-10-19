"""Live Supabase round trip: /predict must log a predictions row linked to a runs row.

Skipped unless SUPABASE_URL and SUPABASE_SERVICE_KEY are set (a local .env counts).
The test deletes the rows it creates.
"""
from __future__ import annotations

import pytest

from tests.conftest import has_supabase_creds, make_image_bytes

pytestmark = pytest.mark.skipif(
    not has_supabase_creds(), reason="No live Supabase credentials in environment."
)


def test_predict_logs_row_linked_to_a_run():
    from fastapi.testclient import TestClient

    from api import db
    from api.main import app

    prediction_id = None
    with TestClient(app) as c:
        try:
            resp = c.post("/predict", data={"model": "improved"},
                          files={"file": ("rt.png", make_image_bytes(48, 48), "image/png")})
            assert resp.status_code == 200
            body = resp.json()
            assert body["logged"] is True, "prediction was not written to Supabase"
            prediction_id = body["prediction_id"]

            sb = db.get_client()
            pred = sb.table("predictions").select("*").eq("id", prediction_id).execute().data
            assert len(pred) == 1
            assert pred[0]["image_hash"] == body["image_sha256"]

            run = sb.table("runs").select("id,model_type").eq("id", pred[0]["served_by_run_id"]).execute().data
            assert len(run) == 1 and run[0]["model_type"] == "improved"

            meta = sb.table("image_metadata").select("*").eq("prediction_id", prediction_id).execute().data
            assert len(meta) == 1 and (meta[0]["width"], meta[0]["height"]) == (48, 48)

            listed = c.get("/predictions", params={"limit": 100}).json()
            row = next(p for p in listed if p["id"] == prediction_id)
            assert row["runs"]["id"] == pred[0]["served_by_run_id"]
        finally:
            if prediction_id is not None:  # cascade also removes the image_metadata row
                db.get_client().table("predictions").delete().eq("id", prediction_id).execute()