"""Pydantic models: the contract between Streamlit, FastAPI, and Supabase."""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Top5Item(BaseModel):
    label: int
    class_name: str
    prob: float


class PredictResponse(BaseModel):
    model_name: str = Field(..., description="'baseline' or 'improved'.")
    run_id: Optional[int] = Field(None, description="runs.id that served this prediction.")
    prediction_id: Optional[int] = None
    logged: bool = Field(..., description="True if the prediction was written to Supabase.")
    label: int
    class_name: str
    confidence: float = Field(..., description="Softmax probability of the top class (uncalibrated).")
    low_confidence: bool = Field(..., description="Top probability below the warning threshold.")
    probs: List[float]
    top5: List[Top5Item]
    image_sha256: str = Field(..., description="SHA-256 of the uploaded bytes. No pixels are stored.")
    width: int
    height: int
    input_png: str = Field(..., description="base64 PNG of the input, at the uploaded size.")
    cam_png: str = Field(..., description="base64 PNG of the Grad-CAM overlay, same size as the input.")


class Run(BaseModel):
    id: int
    name: str
    model_type: str
    dataset: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float = 0.0
    seed: Optional[int] = None
    hyperparams: Dict[str, Any] = Field(default_factory=dict)
    train_loss: Optional[float] = None
    train_acc: Optional[float] = None
    val_loss: Optional[float] = None
    val_acc: Optional[float] = None
    test_loss: Optional[float] = None
    test_acc: Optional[float] = None
    history: Optional[Dict[str, List[float]]] = None
    gallery: Optional[List[Dict[str, Any]]] = None
    weights_uri: str
    created_at: datetime


class RunSummary(BaseModel):
    id: int
    name: str
    model_type: str
    test_acc: Optional[float] = None


class PredictionOut(BaseModel):
    id: int
    image_hash: str
    predicted_label: int
    predicted_class: str
    top5: List[Top5Item]
    served_by_run_id: int
    created_at: datetime
    runs: Optional[RunSummary] = Field(None, description="The run that served this prediction.")


class ClassesResponse(BaseModel):
    classes: List[str]
    n_classes: int


class Health(BaseModel):
    status: str
    model_loader: bool
    supabase: bool


class Version(BaseModel):
    git_sha: str
    torch_version: str
    supabase_project_ref: Optional[str]