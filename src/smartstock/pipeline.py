from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np
import pandas as pd

from smartstock.baseline import add_baseline_predictions
from smartstock.classical import run_classical_benchmark
from smartstock.config import (
    DEFAULT_HORIZONS,
    DEFAULT_RANDOM_SEED,
    FORECASTS_PATH,
    METRICS_PATH,
    MODEL_BUNDLE_PATH,
    PROCESSED_DATA_PATH,
    RAW_DATA_PATH,
    RECOMMENDATIONS_PATH,
)
from smartstock.data import generate_synthetic_retail_data, time_based_split, validate_retail_data
from smartstock.features import build_latest_snapshot, build_model_frame, get_feature_columns
from smartstock.inventory import build_inventory_recommendations
from smartstock.metrics import summarize_metrics
from smartstock.modeling import save_model_bundle, train_models


def _ensure_directories() -> None:
    for path in (RAW_DATA_PATH.parent, PROCESSED_DATA_PATH.parent, MODEL_BUNDLE_PATH.parent, METRICS_PATH.parent):
        path.mkdir(parents=True, exist_ok=True)


def _build_forecast_rows(snapshot: pd.DataFrame, models: dict, feature_columns: list[str], horizons: tuple[int, ...]) -> pd.DataFrame:
    forecast_frames: list[pd.DataFrame] = []
    for horizon in horizons:
        horizon_snapshot = snapshot.copy()
        horizon_snapshot["predicted_demand"] = np.clip(
            models[horizon].predict(horizon_snapshot[feature_columns]),
            a_min=0,
            a_max=None,
        )
        horizon_snapshot["forecast_horizon"] = horizon
        horizon_snapshot["forecast_date"] = horizon_snapshot["date"] + pd.to_timedelta(horizon, unit="D")
        forecast_frames.append(
            horizon_snapshot[
                [
                    "forecast_date",
                    "store_id",
                    "item_id",
                    "category",
                    "predicted_demand",
                    "forecast_horizon",
                    "recent_mean",
                    "recent_std",
                ]
            ]
        )
    return pd.concat(forecast_frames, ignore_index=True)


def bootstrap_demo_artifacts(force: bool = False, include_classical_benchmark: bool = False) -> dict[str, object]:
    _ensure_directories()
    if MODEL_BUNDLE_PATH.exists() and PROCESSED_DATA_PATH.exists() and FORECASTS_PATH.exists() and RECOMMENDATIONS_PATH.exists() and not force:
        with METRICS_PATH.open("r", encoding="utf-8") as file_obj:
            return json.load(file_obj)

    raw_df = generate_synthetic_retail_data(seed=DEFAULT_RANDOM_SEED)
    validate_retail_data(raw_df)
    raw_df.to_csv(RAW_DATA_PATH, index=False)

    feature_frame = build_model_frame(raw_df, horizons=DEFAULT_HORIZONS)
    feature_frame.to_csv(PROCESSED_DATA_PATH, index=False)

    train_df, validation_df, test_df, split = time_based_split(feature_frame)
    feature_columns = get_feature_columns(feature_frame)

    validation_with_baselines = validation_df.copy()
    baseline_metrics: dict[int, dict[str, float]] = {}
    for horizon in DEFAULT_HORIZONS:
        validation_with_baselines = add_baseline_predictions(validation_with_baselines, horizon)
        baseline_metrics[horizon] = summarize_metrics(
            validation_with_baselines.dropna(subset=[f"target_h{horizon}", f"baseline_seasonal_h{horizon}"]),
            f"target_h{horizon}",
            f"baseline_seasonal_h{horizon}",
        )

    models, model_metrics, validation_predictions = train_models(train_df, validation_df, feature_columns, DEFAULT_HORIZONS)
    latest_snapshot = build_latest_snapshot(feature_frame, split.validation_end)
    forecast_frame = _build_forecast_rows(latest_snapshot, models, feature_columns, DEFAULT_HORIZONS)
    forecast_frame.to_csv(FORECASTS_PATH, index=False)

    recommendations = build_inventory_recommendations(forecast_frame)
    recommendations.to_csv(RECOMMENDATIONS_PATH, index=False)

    classical_benchmark = (
        {horizon: run_classical_benchmark(raw_df, horizon) for horizon in DEFAULT_HORIZONS}
        if include_classical_benchmark
        else {horizon: None for horizon in DEFAULT_HORIZONS}
    )

    metrics_payload = {
        "split": {
            "train_end": str(split.train_end.date()),
            "validation_end": str(split.validation_end.date()),
            "test_end": str(split.test_end.date()),
        },
        "baseline_metrics": baseline_metrics,
        "model_metrics": {horizon: asdict(result) for horizon, result in model_metrics.items()},
        "classical_benchmark": classical_benchmark,
        "validation_preview_rows": int(len(validation_predictions)),
    }

    save_model_bundle(
        {
            "models": models,
            "feature_columns": feature_columns,
            "split": {
                "train_end": split.train_end,
                "validation_end": split.validation_end,
                "test_end": split.test_end,
            },
        },
        MODEL_BUNDLE_PATH,
    )

    with METRICS_PATH.open("w", encoding="utf-8") as file_obj:
        json.dump(metrics_payload, file_obj, indent=2)

    return metrics_payload


if __name__ == "__main__":
    payload = bootstrap_demo_artifacts(force=True, include_classical_benchmark=False)
    print(json.dumps(payload, indent=2))
