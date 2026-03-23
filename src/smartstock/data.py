from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {
    "date",
    "store_id",
    "item_id",
    "category",
    "price",
    "promo_flag",
    "event_name",
    "target_sales",
}


@dataclass(frozen=True)
class TimeSplit:
    train_end: pd.Timestamp
    validation_end: pd.Timestamp
    test_end: pd.Timestamp


def generate_synthetic_retail_data(
    n_stores: int = 4,
    n_items: int = 10,
    days: int = 520,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a public-dataset-like retail panel for a runnable demo."""
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2023-01-01")
    dates = pd.date_range(start, periods=days, freq="D")
    categories = ["Produce", "Household", "Beverages", "Snacks", "Frozen"]
    stores = [f"store_{idx + 1}" for idx in range(n_stores)]
    items = [f"item_{idx + 1:02d}" for idx in range(n_items)]
    store_bias = {store: 0.85 + 0.15 * idx for idx, store in enumerate(stores)}

    records: list[dict[str, object]] = []
    for item_index, item in enumerate(items):
        category = categories[item_index % len(categories)]
        base_demand = 14 + (item_index % 5) * 5
        base_price = 5.5 + (item_index % 4) * 1.75
        trend = 1 + item_index * 0.0008
        seasonal_phase = rng.uniform(0, 2 * np.pi)

        for store in stores:
            local_bias = store_bias[store] * rng.uniform(0.9, 1.1)
            for day_idx, date in enumerate(dates):
                day_of_week = date.dayofweek
                month = date.month
                is_weekend = day_of_week >= 5
                promo_flag = int((day_idx + item_index) % 37 in {0, 1, 2})
                holiday_flag = int(date.day in {1, 15, 28} and month in {1, 5, 11, 12})
                event_name = "Holiday" if holiday_flag else ("Promo" if promo_flag else "None")

                annual_seasonality = 1.0 + 0.18 * np.sin((2 * np.pi * day_idx / 365) + seasonal_phase)
                weekly_seasonality = 1.12 if is_weekend else 0.96
                promo_multiplier = 1.35 if promo_flag else 1.0
                holiday_multiplier = 1.28 if holiday_flag else 1.0

                price_noise = rng.normal(0, 0.22)
                price = max(1.0, base_price + price_noise - 0.45 * promo_flag)
                demand_mean = (
                    base_demand
                    * trend
                    * local_bias
                    * annual_seasonality
                    * weekly_seasonality
                    * promo_multiplier
                    * holiday_multiplier
                )
                price_effect = max(0.65, 1.18 - ((price - base_price) * 0.07))
                noise = rng.normal(0, 2.4)
                sales = max(0, demand_mean * price_effect + noise)

                records.append(
                    {
                        "date": date,
                        "store_id": store,
                        "item_id": item,
                        "category": category,
                        "price": round(price, 2),
                        "promo_flag": promo_flag,
                        "event_name": event_name,
                        "target_sales": round(sales, 2),
                    }
                )

    df = pd.DataFrame.from_records(records)
    return df.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)


def validate_retail_data(df: pd.DataFrame) -> None:
    missing_cols = REQUIRED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"Retail dataset is missing required columns: {sorted(missing_cols)}")

    duplicate_mask = df.duplicated(subset=["date", "store_id", "item_id"])
    if duplicate_mask.any():
        raise ValueError("Retail dataset contains duplicate date/store/item combinations.")

    if df["target_sales"].isna().any():
        raise ValueError("Retail dataset contains missing target_sales values.")


def time_based_split(
    df: pd.DataFrame,
    train_fraction: float = 0.7,
    validation_fraction: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, TimeSplit]:
    ordered_dates = pd.Index(sorted(pd.to_datetime(df["date"]).unique()))
    train_idx = max(1, int(len(ordered_dates) * train_fraction))
    validation_idx = max(train_idx + 1, int(len(ordered_dates) * (train_fraction + validation_fraction)))
    validation_idx = min(validation_idx, len(ordered_dates) - 1)

    train_end = ordered_dates[train_idx - 1]
    validation_end = ordered_dates[validation_idx - 1]
    test_end = ordered_dates[-1]

    train_df = df[df["date"] <= train_end].copy()
    validation_df = df[(df["date"] > train_end) & (df["date"] <= validation_end)].copy()
    test_df = df[df["date"] > validation_end].copy()

    split = TimeSplit(
        train_end=pd.Timestamp(train_end),
        validation_end=pd.Timestamp(validation_end),
        test_end=pd.Timestamp(test_end),
    )
    return train_df, validation_df, test_df, split

