"""Compare trained models on the held-out test set.

Loads every run in the MLflow "demand-forecasting" experiment, scores each
model on the same time-based test split used in training, and prints a
metrics table (MAPE / RMSE / MAE).

Usage:
    python src/evaluate.py [--test-days 60]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data_generator import generate_sales
from features import TARGET, build_features, feature_columns
from train import EXPERIMENT, MLRUNS_DIR, mape, rmse, time_split

ROOT = Path(__file__).resolve().parent.parent


def main(test_days: int = 60) -> None:
    mlflow.set_tracking_uri(str(MLRUNS_DIR))
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise SystemExit(
            f"No MLflow experiment '{EXPERIMENT}' found. Run `python src/train.py` first."
        )

    # Rebuild the identical test split.
    sales = generate_sales()
    feat = build_features(sales)
    cols = feature_columns(feat)
    _, test_df = time_split(feat, test_days)
    X_test, y_test = test_df[cols], test_df[TARGET].to_numpy()

    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.test_mape ASC"],
    )
    if runs.empty:
        raise SystemExit("No runs found in the experiment. Run `python src/train.py` first.")

    rows = []
    for _, run in runs.iterrows():
        model = mlflow.pyfunc.load_model(f"runs:/{run['run_id']}/model")
        preds = np.asarray(model.predict(X_test))
        rows.append(
            {
                "run": run["tags.mlflow.runName"],
                "MAPE": mape(y_test, preds),
                "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
                "MAE": float(mean_absolute_error(y_test, preds)),
            }
        )

    table = pd.DataFrame(rows).sort_values("MAPE").reset_index(drop=True)
    print(f"\nTest-set comparison (last {test_days} days, n={len(X_test):,}):\n")
    print(table.to_string(index=False, float_format="%.4f"))
    print(f"\nWinner: {table.iloc[0]['run']} (lowest MAPE)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained models.")
    parser.add_argument("--test-days", type=int, default=60)
    args = parser.parse_args()
    main(test_days=args.test_days)
