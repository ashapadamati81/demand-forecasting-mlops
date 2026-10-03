"""Feature engineering for demand forecasting.

Training pipeline (`build_features`):
  - lag features: units sold 7 / 14 / 28 days ago (per SKU)
  - rolling stats: mean over 7 / 14 / 28 days, std over 7 days (all shifted by
    1 day so no future information leaks into the features)
  - calendar features: day of week, month, day of year, ISO week, quarter,
    weekend flag, holiday flag
  - promo / price: on_promotion flag, current price
  - categorical: one-hot category, integer SKU code

Serving (`build_serving_row`) rebuilds the exact same feature vector from a
single API request: 28 trailing daily sales + date + price/promo/category.
"""

from __future__ import annotations

import pandas as pd

LAG_DAYS = [7, 14, 28]
ROLL_WINDOWS = [7, 14, 28]

TARGET = "units_sold"


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar-derived columns from the ``date`` column (in place copy)."""
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["dow"] = dt.dt.dayofweek.astype(int)          # Mon=0 … Sun=6
    df["month"] = dt.dt.month.astype(int)
    df["dayofyear"] = dt.dt.dayofyear.astype(int)
    df["weekofyear"] = dt.dt.isocalendar().week.astype(int)
    df["quarter"] = dt.dt.quarter.astype(int)
    df["is_weekend"] = (df["dow"] >= 5).astype(int)
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the full modeling table.

    Returns a DataFrame with one row per (sku, date) that has complete
    history (first 28 days per SKU are dropped) and a ``TARGET`` column.
    """
    df = df.sort_values(["sku_id", "date"]).copy()
    df = add_calendar_features(df)
    df["on_promotion"] = df["on_promotion"].astype(int)
    df["is_holiday"] = df["is_holiday"].astype(int)

    grouped = df.groupby("sku_id", sort=False)[TARGET]

    # Lag features: demand N days ago.
    for lag in LAG_DAYS:
        df[f"lag_{lag}"] = grouped.shift(lag)

    # Rolling statistics over past sales only (shift(1) avoids leakage).
    shifted = grouped.shift(1)
    for window in ROLL_WINDOWS:
        df[f"roll_mean_{window}"] = shifted.groupby(
            df["sku_id"], sort=False
        ).rolling(window, min_periods=window).mean().reset_index(level=0, drop=True)
    df["roll_std_7"] = (
        shifted.groupby(df["sku_id"], sort=False)
        .rolling(7, min_periods=7)
        .std()
        .reset_index(level=0, drop=True)
    )

    # SKU identity as an integer code (lets tree models learn per-SKU levels).
    df["sku_code"] = df["sku_id"].astype("category").cat.codes.astype(int)

    # One-hot categories.
    df = pd.get_dummies(df, columns=["category"], prefix="cat", dtype=int)

    # Drop the warm-up rows that lack full 28-day history.
    df = df.dropna().reset_index(drop=True)
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Ordered model input columns (everything except keys and target)."""
    exclude = {"date", "sku_id", TARGET}
    return [c for c in df.columns if c not in exclude]


def build_serving_row(
    *,
    sku_id: str,
    category: str,
    date: str,
    price: float,
    on_promotion: bool,
    sales_last_28: list[float],
    feature_cols: list[str],
    sku_to_code: dict[str, int],
    categories: list[str],
) -> pd.DataFrame:
    """Rebuild one feature row for the /predict endpoint.

    ``sales_last_28``: trailing 28 daily unit sales, oldest → newest.
    ``feature_cols`` / ``sku_to_code`` / ``categories`` come from the
    artifacts saved by the training script so train/serve stay consistent.
    """
    if len(sales_last_28) < 28:
        raise ValueError("sales_last_28 needs at least 28 values")
    sales = list(sales_last_28[-28:])  # keep only the most recent 28
    dt = pd.to_datetime(date)

    row: dict[str, float | int] = {
        "lag_7": sales[-7],
        "lag_14": sales[-14],
        "lag_28": sales[-28],
        "roll_mean_7": sum(sales[-7:]) / 7,
        "roll_mean_14": sum(sales[-14:]) / 14,
        "roll_mean_28": sum(sales) / 28,
        "roll_std_7": float(pd.Series(sales[-7:]).std()),
        "dow": dt.dayofweek,
        "month": dt.month,
        "dayofyear": dt.dayofyear,
        "weekofyear": int(dt.isocalendar().week),
        "quarter": dt.quarter,
        "is_weekend": int(dt.dayofweek >= 5),
        "is_holiday": 0,  # API caller can extend; not required for the demo
        "on_promotion": int(on_promotion),
        "price": float(price),
        # Unknown SKUs get code -1 (tree models treat it as "unseen").
        "sku_code": sku_to_code.get(sku_id, -1),
    }
    for cat in categories:
        row[f"cat_{cat}"] = int(category == cat)

    # Align exactly to the training column order (missing → 0).
    ordered = {col: row.get(col, 0) for col in feature_cols}
    return pd.DataFrame([ordered], columns=feature_cols)
