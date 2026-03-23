from __future__ import annotations

import pandas as pd


def add_baseline_predictions(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    baseline = frame.copy()
    daily_proxy = baseline["rolling_mean_7"].fillna(baseline["lag_7"]).fillna(baseline["lag_1"]).fillna(0.0)
    seasonal_proxy = baseline["rolling_mean_28"].fillna(daily_proxy)
    baseline[f"baseline_naive_h{horizon}"] = daily_proxy * horizon
    baseline[f"baseline_seasonal_h{horizon}"] = seasonal_proxy * horizon
    return baseline

