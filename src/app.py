from __future__ import annotations

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from src.data_processing import add_features
from src.infrastructure_monitoring import get_infrastructure_metrics


app = FastAPI(title="Diamonds Price Prediction API", version="1.0.0")


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


@app.get("/", response_model=RootResponse)
def root() -> RootResponse:
    return {"message": "Diamonds price prediction API", "docs": "/docs"}


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return {
        "status": "ok",
        "model_loaded": False,
        "infrastructure": get_infrastructure_metrics(),
    }
