from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.data_processing import add_features
from src.model_training import FEATURE_COLUMNS, METRICS_PATH, MODEL_PATH, load_model
from src.monitoring import get_infrastructure_metrics


app = FastAPI(title="Diamonds Price Prediction API", version="1.0.0")
model: Any | None = load_model()


class InfrastructureMetrics(BaseModel):
    cpu_percent: float
    ram_percent: float
    disk_percent: float


class RootResponse(BaseModel):
    message: str
    docs: str


class HealthResponse(BaseModel):
    status: Literal["ok"]
    model_loaded: bool
    infrastructure: InfrastructureMetrics


class DiamondFeatures(BaseModel):
    carat: float = Field(gt=0, examples=[1.0])
    cut: str = Field(min_length=1, examples=["Ideal"])
    color: str = Field(min_length=1, examples=["E"])
    clarity: str = Field(min_length=1, examples=["VS1"])
    depth: float = Field(gt=0, examples=[61.0])
    table: float = Field(gt=0, examples=[57.0])
    x: float = Field(gt=0, examples=[6.0])
    y: float = Field(gt=0, examples=[6.1])
    z: float = Field(gt=0, examples=[3.8])

    model_config = {
        "json_schema_extra": {
            "example": {
                "carat": 1.0,
                "cut": "Ideal",
                "color": "E",
                "clarity": "VS1",
                "depth": 61.0,
                "table": 57.0,
                "x": 6.0,
                "y": 6.1,
                "z": 3.8,
            }
        }
    }


class PredictionResponse(BaseModel):
    predicted_price: float
    message: str


class ModelInfoResponse(BaseModel):
    model_path: str
    model_exists: bool
    metrics: dict[str, float] | None
    features: list[str]
    message: str


def _get_model() -> Any:
    global model
    if model is None:
        model = load_model()
    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not trained yet. Run `python -m src.model_training` before calling /predict.",
        )
    return model


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return {"message": "Diamonds price prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {
        "status": "ok",
        "model_loaded": model is not None or MODEL_PATH.exists(),
        "infrastructure": get_infrastructure_metrics(),
    }


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        503: {
            "description": "The model artifact is not available yet.",
            "content": {
                "application/json": {
                    "example": {
                        "detail": "Model is not trained yet. Run `python -m src.model_training` before calling /predict."
                    }
                }
            },
        }
    },
)
def predict(features: DiamondFeatures) -> PredictionResponse:
    active_model = _get_model()
    payload = features.model_dump() if hasattr(features, "model_dump") else features.dict()
    frame = pd.DataFrame([payload])
    frame = add_features(frame)
    try:
        prediction = float(active_model.predict(frame[FEATURE_COLUMNS])[0])
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Prediction failed. Check input values and model artifact compatibility.",
        ) from exc
    return {
        "predicted_price": round(prediction, 2),
        "message": "Prediction completed successfully.",
    }


@app.get("/model/info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    metrics: dict[str, Any] | None = None
    if Path(METRICS_PATH).exists():
        metrics = json.loads(Path(METRICS_PATH).read_text(encoding="utf-8"))
    model_exists = MODEL_PATH.exists()
    return {
        "model_path": str(MODEL_PATH),
        "model_exists": model_exists,
        "metrics": metrics,
        "features": FEATURE_COLUMNS,
        "message": "Model artifact found." if model_exists else "Model artifact is missing. Run training before prediction.",
    }
