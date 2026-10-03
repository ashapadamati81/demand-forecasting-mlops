# Demand Forecasting MLOps

Retail demand forecasting, end to end: synthetic data generation → feature
engineering → experiment tracking with MLflow → model serving with FastAPI.

## Problem statement

A retailer needs next-day demand forecasts per SKU to drive replenishment and
staffing. This project builds a complete, reproducible ML pipeline on
**synthetic** daily sales data (2 years, ~50 SKUs) that exhibits the structure
real retail data has: weekly and yearly seasonality, trend, promotions, and
holiday effects.

## Approach

1. **Synthetic data** (`src/data_generator.py`) — 2 years of daily unit sales
   for 50 SKUs across 4 categories, with weekly/yearly seasonality, per-SKU
   trend, random promotion events (10–35% discounts, 1.4–2.2× demand lift),
   US holidays, and log-normal noise. Fully seeded and reproducible.
2. **Feature engineering** (`src/features.py`) — lag features (7/14/28 days),
   rolling means/std (shifted by 1 day to prevent leakage), calendar features
   (day of week, month, day of year, ISO week, quarter, weekend, holiday),
   promotion flag, price, one-hot category, and SKU code.
3. **Experiments** (`src/train.py`) — time-based split (most recent 60 days
   held out), 3 tracked runs in MLflow: XGBoost baseline, tuned XGBoost, and
   a LightGBM baseline. Params, metrics (MAPE/RMSE/MAE), and models are all
   logged to a local file store. The best run (lowest test MAPE) is saved as
   the champion with its serving artifacts.
4. **Evaluation** (`src/evaluate.py`) — reloads every run from MLflow and
   prints a head-to-head metrics table on the held-out set.
5. **Serving** (`app/main.py`) — FastAPI app loading the champion model via
   MLflow pyfunc; `POST /predict` takes 28 days of trailing sales + date,
   price, promo flag, and category, and returns next-day predicted units with
   pydantic request validation.

## Project structure

```
demand-forecasting-mlops/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── data_generator.py   # synthetic sales panel generator
│   ├── features.py         # lag/rolling/calendar/promo features (+ serving helper)
│   ├── train.py            # 3 MLflow experiments + champion selection
│   └── evaluate.py         # head-to-head model comparison on test set
├── app/
│   └── main.py             # FastAPI server: GET /health, POST /predict
├── data/                   # generated sales.csv (git-ignored)
└── models/                 # champion URI + serving artifacts (written by train.py)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## How to train

```bash
python src/train.py                 # 3 experiments, ~1-2 min
python src/train.py --test-days 90  # longer holdout (optional)
```

Browse the experiments:

```bash
mlflow ui   # → http://127.0.0.1:5000
```

## How to evaluate

```bash
python src/evaluate.py
```

## How to run the API

```bash
# from the project root, after training
uvicorn app.main:app --reload --port 8000
```

Example request (predict next-day units for SKU-001):

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sku_id": "SKU-001",
    "category": "grocery",
    "date": "2025-01-15",
    "price": 12.99,
    "on_promotion": false,
    "sales_last_28": [42, 38, 45, 51, 47, 39, 44, 40, 46, 52, 48, 41, 43, 49,
                      55, 50, 44, 39, 42, 47, 53, 51, 46, 40, 38, 45, 49, 52]
  }'
```

Example response:

```json
{
  "sku_id": "SKU-001",
  "date": "2025-01-15",
  "predicted_units": 46.83,
  "model_uri": "runs:/<run-id>/model"
}
```

## Sample results

Test set = most recent 60 days (3,000 SKU-days), metrics from `python src/evaluate.py`:

| run           | MAPE   | RMSE  | MAE   |
|---------------|--------|-------|-------|
| xgb_baseline  | 0.2285 | 35.59 | 22.57 |
| lgbm_baseline | 0.2318 | 35.51 | 22.83 |
| xgb_tuned     | 0.2346 | 35.46 | 22.84 |

*From the reference run on this machine (seed 42) — rerun `train.py` /
`evaluate.py` to reproduce. Small variations across library versions are
normal.*

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · LightGBM · MLflow (local
file store) · FastAPI · uvicorn · pydantic
