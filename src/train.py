"""Train demand forecasting models with MLflow experiment tracking.

Runs 3 experiments on a time-based split (last ``--test-days`` days held out):
  1. xgb_baseline — XGBoost with default-ish hyperparameters
  2. xgb_tuned    — XGBoost with tuned hyperparameters
  3. lgbm_baseline — LightGBM baseline for comparison

Each run logs hyperparameters, test metrics (MAPE / RMSE / MAE) and the
trained model to a local MLflow file store (./mlruns). The best run (lowest
test MAPE) is recorded to models/best_model_uri.txt along with the serving
artifacts (feature column order, SKU code mapping, category list) that the
FastAPI app needs.

Usage:
    python src/train.py [--test-days 60] [--seed 42]
    mlflow ui   # then open http://127.0.0.1:5000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_generator import CATEGORIES, generate_sales
from features import TARGET, build_features, feature_columns

# Project root = parent of src/
ROOT = Path(__file__).resolve().parent.parent
MLRUNS_DIR = ROOT / "mlruns"
DATA_PATH = ROOT / "data" / "sales.csv"
MODELS_DIR = ROOT / "models"
EXPERIMENT = "demand-forecasting"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute percentage error; ignores zero actuals (no division by 0)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def time_split(df: pd.DataFrame, test_days: int):
    """Split by date: the most recent ``test_days`` are the test set."""
    cutoff = df["date"].max() - pd.Timedelta(days=test_days - 1)
    train = df[df["date"] < cutoff].reset_index(drop=True)
    test = df[df["date"] >= cutoff].reset_index(drop=True)
    return train, test


def run_experiment(
    name: str,
    make_model,
    flavor: str,
    params: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[str, float]:
    """Train one model, log everything to MLflow. Returns (run_id, test MAPE)."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.log_params(params)
        model = make_model(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        metrics = {
            "test_mape": mape(y_test.to_numpy(), preds),
            "test_rmse": rmse(y_test.to_numpy(), preds),
            "test_mae": float(mean_absolute_error(y_test, preds)),
        }
        mlflow.log_metrics(metrics)

        # Log the model in its native flavor; served later via pyfunc.
        if flavor == "xgboost":
            mlflow.xgboost.log_model(model, artifact_path="model")
        elif flavor == "lightgbm":
            mlflow.lightgbm.log_model(model, artifact_path="model")
        else:
            raise ValueError(f"unknown flavor: {flavor}")

        print(f"[{name}] MAPE={metrics['test_mape']:.4f} "
              f"RMSE={metrics['test_rmse']:.2f} MAE={metrics['test_mae']:.2f}")
        return run.info.run_id, metrics["test_mape"]


def main(test_days: int = 60, seed: int = 42) -> None:
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    mlflow.set_experiment(EXPERIMENT)
    MODELS_DIR.mkdir(exist_ok=True)
    (ROOT / "data").mkdir(exist_ok=True)

    # 1. Data ------------------------------------------------------------
    print("Generating synthetic sales data…")
    sales = generate_sales(seed=seed)
    sales.to_csv(DATA_PATH, index=False)
    print(f"  saved {sales.shape[0]:,} rows → {DATA_PATH}")

    # 2. Features ---------------------------------------------------------
    print("Engineering features…")
    feat = build_features(sales)
    cols = feature_columns(feat)
    X = feat[cols]
    y = feat[TARGET]
    train_df, test_df = time_split(feat, test_days)
    X_train, y_train = train_df[cols], train_df[TARGET]
    X_test, y_test = test_df[cols], test_df[TARGET]
    print(f"  features={len(cols)}, train={len(train_df):,}, test={len(test_df):,}")

    # 3. Experiments ------------------------------------------------------
    experiments = [
        (
            "xgb_baseline",
            xgb.XGBRegressor,
            "xgboost",
            dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                 random_state=seed, n_jobs=-1),
        ),
        (
            "xgb_tuned",
            xgb.XGBRegressor,
            "xgboost",
            dict(n_estimators=800, max_depth=8, learning_rate=0.05,
                 subsample=0.8, colsample_bytree=0.8,
                 random_state=seed, n_jobs=-1),
        ),
        (
            "lgbm_baseline",
            lgb.LGBMRegressor,
            "lightgbm",
            dict(n_estimators=500, learning_rate=0.05, num_leaves=63,
                 random_state=seed, n_jobs=-1, verbose=-1),
        ),
    ]

    results: list[tuple[str, str, float]] = []
    for name, cls, flavor, params in experiments:
        run_id, test_mape = run_experiment(
            name, cls, flavor, params, X_train, y_train, X_test, y_test
        )
        results.append((name, run_id, test_mape))

    # 4. Champion selection ------------------------------------------------
    best_name, best_run_id, best_mape = min(results, key=lambda r: r[2])
    print(f"\nBest model: {best_name} (test MAPE={best_mape:.4f})")

    # Serving artifacts the FastAPI app needs to reproduce the feature space.
    (MODELS_DIR / "best_model_uri.txt").write_text(f"runs:/{best_run_id}/model")
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(cols, indent=2))
    sku_to_code = {
        sku: int(code)
        for code, sku in enumerate(sorted(sales["sku_id"].unique()))
    }
    # NOTE: build_features() codes SKUs via pandas Categorical on the full
    # frame; sorted-unique enumeration reproduces the same mapping.
    (MODELS_DIR / "sku_to_code.json").write_text(json.dumps(sku_to_code, indent=2))
    (MODELS_DIR / "categories.json").write_text(json.dumps(CATEGORIES, indent=2))
    print(f"  wrote serving artifacts → {MODELS_DIR}/")

    # Sanity check the mapping matches what build_features produced.
    check = dict(zip(feat["sku_id"].astype("category").cat.categories,
                     range(feat["sku_id"].nunique())))
    assert all(sku_to_code[k] == v for k, v in check.items()), "SKU code mismatch!"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand forecasting models.")
    parser.add_argument("--test-days", type=int, default=60,
                        help="Hold out the most recent N days for testing.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()
    main(test_days=args.test_days, seed=args.seed)
