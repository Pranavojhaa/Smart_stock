from __future__ import annotations

import pandas as pd


def _future_sum_by_group(series: pd.Series, horizon: int) -> pd.Series:
    return series.shift(-1).rolling(window=horizon, min_periods=horizon).sum().shift(-(horizon - 1))


def build_model_frame(df: pd.DataFrame, horizons: tuple[int, ...] = (7, 28)) -> pd.DataFrame:
    frame = df.copy()
    frame["date"] = pd.to_datetime(frame["date"])
    frame = frame.sort_values(["store_id", "item_id", "date"]).reset_index(drop=True)

    frame["day_of_week"] = frame["date"].dt.dayofweek
    frame["week_of_year"] = frame["date"].dt.isocalendar().week.astype(int)
    frame["month"] = frame["date"].dt.month
    frame["day_of_month"] = frame["date"].dt.day
    frame["is_weekend"] = (frame["day_of_week"] >= 5).astype(int)
    frame["is_holiday"] = frame["event_name"].eq("Holiday").astype(int)
    frame["promo_flag"] = frame["promo_flag"].astype(int)

    for lag in (1, 7, 14, 28):
        frame[f"lag_{lag}"] = frame.groupby(["store_id", "item_id"])["target_sales"].shift(lag)

    for window in (7, 14, 28):
        shifted = frame.groupby(["store_id", "item_id"])["target_sales"].shift(1)
        frame[f"rolling_mean_{window}"] = (
            shifted.groupby([frame["store_id"], frame["item_id"]]).rolling(window, min_periods=window).mean().reset_index(level=[0, 1], drop=True)
        )
        frame[f"rolling_std_{window}"] = (
            shifted.groupby([frame["store_id"], frame["item_id"]]).rolling(window, min_periods=window).std().reset_index(level=[0, 1], drop=True)
        )

    category_shifted = frame.groupby(["store_id", "category"])["target_sales"].shift(1)
    frame["category_rolling_mean_14"] = (
        category_shifted.groupby([frame["store_id"], frame["category"]]).rolling(14, min_periods=14).mean().reset_index(level=[0, 1], drop=True)
    )
    frame["store_rolling_mean_14"] = (
        frame.groupby("store_id")["target_sales"].shift(1).groupby(frame["store_id"]).rolling(14, min_periods=14).mean().reset_index(level=0, drop=True)
    )

    for horizon in horizons:
        frame[f"target_h{horizon}"] = frame.groupby(["store_id", "item_id"])["target_sales"].transform(
            lambda series, h=horizon: _future_sum_by_group(series, h)
        )

    frame["recent_std"] = frame["rolling_std_28"].fillna(frame["rolling_std_14"]).fillna(0.0)
    frame["recent_mean"] = frame["rolling_mean_28"].fillna(frame["rolling_mean_14"]).fillna(frame["rolling_mean_7"])

    return frame


def get_feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "date",
        "store_id",
        "item_id",
        "category",
        "event_name",
        "target_sales",
    }
    return [column for column in frame.columns if column not in excluded and not column.startswith("target_h")]


def build_latest_snapshot(frame: pd.DataFrame, split_date: pd.Timestamp) -> pd.DataFrame:
    snapshot = frame[frame["date"] <= split_date].copy()
    snapshot = snapshot.dropna().sort_values(["store_id", "item_id", "date"])
    snapshot = snapshot.groupby(["store_id", "item_id"], as_index=False).tail(1)
    return snapshot.reset_index(drop=True)
