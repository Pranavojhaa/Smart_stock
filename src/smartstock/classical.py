from __future__ import annotations

import pandas as pd


def run_classical_benchmark(df: pd.DataFrame, horizon: int) -> dict[str, float] | None:
    """Optional aggregate SARIMAX benchmark when statsmodels is available."""
    try:
        from statsmodels.tsa.statespace.sarimax import SARIMAX
    except Exception:
        return None

    aggregate = df.groupby("date", as_index=False)["target_sales"].sum().sort_values("date")
    if len(aggregate) <= horizon + 30:
        return None

    train_series = aggregate["target_sales"].iloc[:-horizon]
    test_series = aggregate["target_sales"].iloc[-horizon:]

    model = SARIMAX(
        train_series,
        order=(1, 0, 1),
        seasonal_order=(1, 0, 1, 7),
        trend="c",
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    forecast = fit.forecast(horizon)
    return {
        "actual_total": float(test_series.sum()),
        "predicted_total": float(forecast.sum()),
    }

