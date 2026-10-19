"""Supabase helpers for the API tier (service-role key; server-side only).

Stored per prediction: SHA-256 hash, dimensions, predicted label, top-5
probabilities, and the serving run. Never pixels.

Env vars: SUPABASE_URL, SUPABASE_SERVICE_KEY
"""
import os
from typing import List, Optional

from dotenv import load_dotenv
from supabase import Client, create_client

load_dotenv()  # local dev: read .env; on Render the vars come from the dashboard

_client: Optional[Client] = None

_client: Optional[Client] = None

_PRED_COLS = (
    "id,image_hash,predicted_label,predicted_class,top5,served_by_run_id,created_at,"
    "runs(id,name,model_type,test_acc)"
)


def get_client() -> Client:
    global _client
    if _client is None:
        _client = create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SERVICE_KEY"])
    return _client


def ping() -> bool:
    try:
        get_client().table("runs").select("id").limit(1).execute()
        return True
    except Exception:
        return False


def get_run(run_id: int) -> Optional[dict]:
    resp = get_client().table("runs").select("*").eq("id", run_id).limit(1).execute()
    return resp.data[0] if resp.data else None


def get_run_for_model(model_type: str) -> Optional[dict]:
    """Newest run of the given model type ('baseline' or 'improved')."""
    resp = (
        get_client().table("runs").select("*")
        .eq("model_type", model_type).order("created_at", desc=True).limit(1).execute()
    )
    return resp.data[0] if resp.data else None


def latest_runs(limit: int = 20) -> List[dict]:
    return (
        get_client().table("runs").select("*")
        .order("created_at", desc=True).limit(limit).execute().data
    )


def insert_prediction(
    *, image_hash: str, width: int, height: int, channels: int,
    label: int, class_name: str, top5: List[dict], run_id: int,
) -> dict:
    """Write a predictions row plus its image_metadata row. If the metadata
    insert fails, the prediction row is removed so the two tables stay in sync."""
    client = get_client()
    pred = client.table("predictions").insert({
        "image_hash": image_hash,
        "predicted_label": label,
        "predicted_class": class_name,
        "top5": top5,
        "served_by_run_id": run_id,
    }).execute().data[0]
    try:
        client.table("image_metadata").insert({
            "prediction_id": pred["id"],
            "image_hash": image_hash,
            "width": width,
            "height": height,
            "channels": channels,
            "label": class_name,
        }).execute()
    except Exception:
        client.table("predictions").delete().eq("id", pred["id"]).execute()
        raise
    return pred


def latest_predictions(limit: int = 50) -> List[dict]:
    """Recent predictions joined to the run that served them."""
    return (
        get_client().table("predictions").select(_PRED_COLS)
        .order("created_at", desc=True).limit(limit).execute().data
    )