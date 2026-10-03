"""Demand forecasting model server.

Loads the champion model logged by ``src/train.py`` (via MLflow) and serves
next-day demand predictions.

Run from the project root (after training):
    uvicorn app.main:app --reload --port 8000

Then:
    curl -X POST http://127.0.0.1:8000/predict -H "Content-Type: application/json" -d '{...}'
(see README for a full example)
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import date as date_type
from pathlib import Path
from typing import Any

import mlflow
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.features import build_serving_row

# Project root = parent of app/
ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"

# Loaded once at startup.
_model: Any = None
_feature_cols: list[str] = []
_sku_to_code: dict[str, int] = {}
_categories: list[str] = []


class PredictRequest(BaseModel):
    """One next-day forecast request.

    sales_last_28: trailing 28 daily unit sales for the SKU, oldest → newest.
    """

    sku_id: str = Field(..., examples=["SKU-001"])
    category: str = Field(..., examples=["grocery"])
    date: date_type = Field(..., description="Forecast date (YYYY-MM-DD)")
    price: float = Field(..., gt=0, examples=[29.99])
    on_promotion: bool = Field(False)
    sales_last_28: list[float] = Field(
        ..., min_length=28, description="Trailing 28 daily unit sales"
    )


class PredictResponse(BaseModel):
    sku_id: str
    date: str
    predicted_units: float
    model_uri: str


def load_artifacts() -> None:
    """Load the champion model + serving artifacts written by train.py."""
    global _model, _feature_cols, _sku_to_code, _categories

    uri_file = MODELS_DIR / "best_model_uri.txt"
    if not uri_file.exists():
        raise RuntimeError(
            f"{uri_file} not found — run `python src/train.py` first to train "
            "a model and select the champion."
        )
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    model_uri = uri_file.read_text().strip()
    _model = mlflow.pyfunc.load_model(model_uri)
    _feature_cols = json.loads((MODELS_DIR / "feature_columns.json").read_text())
    _sku_to_code = json.loads((MODELS_DIR / "sku_to_code.json").read_text())
    _categories = json.loads((MODELS_DIR / "categories.json").read_text())
    load_artifacts.model_uri = model_uri  # type: ignore[attr-defined]


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_artifacts()
    yield


app = FastAPI(
    title="Demand Forecasting API",
    description="Next-day retail demand predictions from the MLflow champion model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_uri": getattr(load_artifacts, "model_uri", "not loaded")}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if req.category not in _categories:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown category '{req.category}'. Known: {_categories}",
        )
    try:
        row = build_serving_row(
            sku_id=req.sku_id,
            category=req.category,
            date=req.date.isoformat(),
            price=req.price,
            on_promotion=req.on_promotion,
            sales_last_28=req.sales_last_28,
            feature_cols=_feature_cols,
            sku_to_code=_sku_to_code,
            categories=_categories,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    pred = float(_model.predict(row)[0])
    return PredictResponse(
        sku_id=req.sku_id,
        date=req.date.isoformat(),
        predicted_units=max(0.0, round(pred, 2)),  # demand can't be negative
        model_uri=getattr(load_artifacts, "model_uri", ""),
    )
