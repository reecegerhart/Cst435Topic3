"""FastAPI model service: Cloud #2 (Render.com).

Serves the trained CNNs, returns Grad-CAM overlays, and logs predictions
(hash + dimensions + label only) to Supabase.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import subprocess
from contextlib import asynccontextmanager
from typing import Literal, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from api import db
from api.gradcam import get_model, predict_with_cam
from shared.data import CLASS_NAMES, N_CLASSES, to_gray_28
from shared.schemas import (
    ClassesResponse, Health, PredictionOut, PredictResponse, Run, Version,
)

log = logging.getLogger("see-sense")

MODEL_NAMES = ("baseline", "improved")
DEFAULT_MODEL = "improved"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_SIDE = 1024            # px; larger uploads are rejected
LOW_CONFIDENCE = 0.60      # below this the UI should warn the user

TAGS = [
    {"name": "prediction", "description": "Classify an uploaded image and get a Grad-CAM overlay."},
    {"name": "runs", "description": "Training runs (hyperparameters, metrics, weights pointer) from Supabase."},
    {"name": "predictions", "description": "Recent anonymized predictions joined to the run that served them."},
    {"name": "meta", "description": "Class labels."},
    {"name": "ops", "description": "Health and version."},
]


@asynccontextmanager
async def lifespan(_app: FastAPI):
    for name in MODEL_NAMES:  # warm the cache so the first request is fast
        try:
            get_model(name)
        except Exception:
            log.exception("could not load model %s", name)
    yield


app = FastAPI(
    title="See-Sense API",
    description="Fashion-MNIST CNN classifier with Grad-CAM. Baseline vs. improved model.",
    version="2.0.0",
    openapi_tags=TAGS,
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("ALLOWED_ORIGINS", "*").split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


def _git_sha() -> str:
    if os.environ.get("RENDER_GIT_COMMIT"):
        return os.environ["RENDER_GIT_COMMIT"][:7]
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        return "unknown"


def _project_ref() -> Optional[str]:
    url = os.environ.get("SUPABASE_URL", "")
    return url.split("//", 1)[1].split(".", 1)[0] if url.startswith("https://") else None


def _db(fn, *args):
    """Run a Supabase call; surface failures as 502 instead of a bare 500."""
    try:
        return fn(*args)
    except Exception as exc:
        log.exception("Supabase call failed")
        raise HTTPException(status_code=502, detail=f"database error: {type(exc).__name__}")


# ---------------------------------------------------------------------------
# prediction
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=PredictResponse, tags=["prediction"])
def predict(
    file: UploadFile = File(..., description="PNG or JPG, up to 5 MB and 1024 px per side."),
    model: Literal["baseline", "improved"] = Form(DEFAULT_MODEL),
) -> PredictResponse:
    """Classify an uploaded image; return label, probabilities, and a Grad-CAM overlay.

    The overlay has the same dimensions as the uploaded image. Only the image's
    SHA-256 hash, dimensions, and predicted label are logged, never pixels.
    Logging is best-effort: if Supabase is unreachable the prediction is still
    returned with `logged=false`.
    """
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if not raw:
        raise HTTPException(status_code=422, detail="empty file")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="file larger than 5 MB")
    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
    except Exception:
        raise HTTPException(status_code=422, detail="could not decode image")

    width, height = img.size
    if max(width, height) > MAX_SIDE:
        raise HTTPException(status_code=422, detail=f"image larger than {MAX_SIDE}px per side")
    channels = len(img.getbands())

    try:
        result = predict_with_cam(model, to_gray_28(img), out_size=(width, height))
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail=f"weights for '{model}' not found on server")

    sha256 = hashlib.sha256(raw).hexdigest()
    run_id = prediction_id = None
    logged = False
    try:
        run = db.get_run_for_model(model)
        if run is None:
            raise LookupError(f"no runs row for model_type={model}")
        pred = db.insert_prediction(
            image_hash=sha256, width=width, height=height, channels=channels,
            label=result["label"], class_name=result["class_name"],
            top5=result["top5"], run_id=run["id"],
        )
        run_id, prediction_id, logged = run["id"], pred["id"], True
    except Exception:
        log.exception("could not log prediction")

    return PredictResponse(
        run_id=run_id, prediction_id=prediction_id, logged=logged,
        low_confidence=result["confidence"] < LOW_CONFIDENCE,
        image_sha256=sha256, width=width, height=height,
        **{k: result[k] for k in (
            "model_name", "label", "class_name", "confidence",
            "probs", "top5", "input_png", "cam_png")},
    )


# ---------------------------------------------------------------------------
# runs / predictions
# ---------------------------------------------------------------------------
@app.get("/runs", response_model=list[Run], tags=["runs"])
def list_runs(limit: int = Query(20, ge=1, le=100)) -> list[Run]:
    """Most recent training runs."""
    return [Run(**r) for r in _db(db.latest_runs, limit)]


@app.get("/runs/{run_id}", response_model=Run, tags=["runs"])
def get_run(run_id: int) -> Run:
    row = _db(db.get_run, run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return Run(**row)


@app.get("/predictions", response_model=list[PredictionOut], tags=["predictions"])
def list_predictions(limit: int = Query(50, ge=1, le=100)) -> list[PredictionOut]:
    """Recent predictions (hash, label, top-5), each joined to its serving run."""
    return [PredictionOut(**p) for p in _db(db.latest_predictions, limit)]


# ---------------------------------------------------------------------------
# meta / ops
# ---------------------------------------------------------------------------
@app.get("/classes", response_model=ClassesResponse, tags=["meta"])
def classes() -> ClassesResponse:
    return ClassesResponse(classes=CLASS_NAMES, n_classes=N_CLASSES)


@app.get("/healthz", response_model=Health, tags=["ops"])
def healthz() -> Health:
    supabase_ok = db.ping()
    try:
        for name in MODEL_NAMES:
            get_model(name)
        model_ok = True
    except Exception:
        model_ok = False
    return Health(
        status="ok" if (supabase_ok and model_ok) else "degraded",
        model_loader=model_ok,
        supabase=supabase_ok,
    )


@app.get("/version", response_model=Version, tags=["ops"])
def version() -> Version:
    return Version(
        git_sha=_git_sha(),
        torch_version=torch.__version__,
        supabase_project_ref=_project_ref(),
    )