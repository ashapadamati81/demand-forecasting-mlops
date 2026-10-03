"""Synthetic retail sales data generator.

Produces 2 years of daily unit sales for ~50 SKUs with realistic structure:
  - per-SKU base demand, price, and trend
  - weekly seasonality (weekend uplift varies by category)
  - yearly seasonality (holiday-season uplift in Nov/Dec)
  - random promotion events with demand uplift + price discount
  - US public holidays with demand uplift
  - multiplicative noise, clipped to non-negative integer units

Everything is seeded for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fixed US public holidays (month, day) that lift demand in this demo.
HOLIDAYS = [
    (1, 1),    # New Year's Day
    (5, 27),   # Memorial Day (approx)
    (7, 4),    # Independence Day
    (9, 2),    # Labor Day (approx)
    (11, 28),  # Thanksgiving (approx)
    (12, 25),  # Christmas
    (12, 31),  # New Year's Eve
]

CATEGORIES = ["grocery", "apparel", "electronics", "home"]

# Weekend uplift factor per category (1.0 = no weekend effect).
WEEKEND_UPLIFT = {
    "grocery": 1.35,
    "apparel": 1.25,
    "electronics": 1.10,
    "home": 1.15,
}


def generate_sales(
    n_skus: int = 50,
    start_date: str = "2023-01-01",
    days: int = 730,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic daily sales panel.

    Returns a DataFrame with columns:
      date, sku_id, category, units_sold, price, on_promotion, is_holiday
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range(start=start_date, periods=days, freq="D")

    # Per-SKU attributes drawn once.
    sku_ids = [f"SKU-{i:03d}" for i in range(1, n_skus + 1)]
    categories = rng.choice(CATEGORIES, size=n_skus)
    base_demand = rng.uniform(8, 120, size=n_skus)          # avg daily units
    base_price = rng.uniform(4.99, 199.99, size=n_skus).round(2)
    trend = rng.uniform(-0.0004, 0.0008, size=n_skus)       # daily growth rate

    holiday_set = set(HOLIDAYS)
    is_holiday_day = np.array(
        [(d.month, d.day) in holiday_set for d in dates]
    )
    day_of_week = dates.dayofweek.to_numpy()               # Mon=0 … Sun=6
    is_weekend = (day_of_week >= 5).astype(float)
    day_of_year = dates.dayofyear.to_numpy()

    # Yearly seasonality: smooth peak around mid-December (holiday shopping).
    yearly = 1.0 + 0.35 * np.exp(-((day_of_year - 350) ** 2) / (2 * 25**2))
    # Small summer dip.
    yearly *= 1.0 - 0.10 * np.exp(-((day_of_year - 200) ** 2) / (2 * 30**2))

    frames = []
    for i, sku in enumerate(sku_ids):
        cat = categories[i]
        t = np.arange(days)

        # Deterministic demand components.
        demand = base_demand[i] * (1.0 + trend[i]) ** t
        demand *= 1.0 + (WEEKEND_UPLIFT[cat] - 1.0) * is_weekend
        demand *= yearly
        demand *= np.where(is_holiday_day, 1.45, 1.0)  # holiday uplift

        # Random promotion events: ~8 promos/year, each 3–10 days long.
        on_promo = np.zeros(days, dtype=bool)
        price = np.full(days, base_price[i])
        n_promos = rng.integers(12, 20)
        for _ in range(n_promos):
            start = rng.integers(0, days - 3)
            length = rng.integers(3, 11)
            end = min(days, start + length)
            discount = rng.uniform(0.10, 0.35)  # 10–35% off
            on_promo[start:end] = True
            price[start:end] = round(base_price[i] * (1 - discount), 2)
        demand *= np.where(on_promo, rng.uniform(1.4, 2.2), 1.0)  # promo lift

        # Multiplicative noise; clip at zero and round to whole units.
        noise = rng.lognormal(mean=0.0, sigma=0.25, size=days)
        units = np.maximum(0, np.round(demand * noise)).astype(int)

        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "sku_id": sku,
                    "category": cat,
                    "units_sold": units,
                    "price": price,
                    "on_promotion": on_promo,
                    "is_holiday": is_holiday_day,
                }
            )
        )

    df = pd.concat(frames, ignore_index=True)
    return df.sort_values(["sku_id", "date"]).reset_index(drop=True)


if __name__ == "__main__":
    # Quick sanity check when run directly.
    df = generate_sales()
    print(df.head(8).to_string(index=False))
    print(f"\nshape={df.shape}, date range={df['date'].min().date()} → {df['date'].max().date()}")
    print(f"promo-day share={df['on_promotion'].mean():.1%}, SKUs={df['sku_id'].nunique()}")
