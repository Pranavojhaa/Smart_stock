from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

from smartstock.metrics import summarize_metrics


@dataclass
class HorizonModelResult:
    horizon: int
    model_name: str
    metrics: dict[str, float]


def train_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    feature_columns: list[str],
    horizons: tuple[int, ...] = (7, 28),
) -> tuple[dict[int, GradientBoostingRegressor], dict[int, HorizonModelResult], pd.DataFrame]:
    models: dict[int, GradientBoostingRegressor] = {}
    metrics: dict[int, HorizonModelResult] = {}
    validation_predictions: list[pd.DataFrame] = []

    for horizon in horizons:
        target_column = f"target_h{horizon}"
        train_subset = train_df.dropna(subset=feature_columns + [target_column]).copy()
        validation_subset = validation_df.dropna(subset=feature_columns + [target_column]).copy()

        model = GradientBoostingRegressor(
            learning_rate=0.05,
            max_depth=4,
            n_estimators=250,
            subsample=0.9,
            random_state=42,
        )
        model.fit(train_subset[feature_columns], train_subset[target_column])
        validation_subset[f"prediction_h{horizon}"] = np.clip(
            model.predict(validation_subset[feature_columns]),
            a_min=0,
            a_max=None,
        )

        metric_summary = summarize_metrics(validation_subset, target_column, f"prediction_h{horizon}")
        models[horizon] = model
        metrics[horizon] = HorizonModelResult(horizon, "GradientBoostingRegressor", metric_summary)
        validation_predictions.append(validation_subset[["date", "store_id", "item_id", target_column, f"prediction_h{horizon}"]])

    combined_predictions = pd.concat(validation_predictions, axis=1)
    combined_predictions = combined_predictions.loc[:, ~combined_predictions.columns.duplicated()]
    return models, metrics, combined_predictions


def save_model_bundle(bundle: dict, destination) -> None:
    joblib.dump(bundle, destination)


def load_model_bundle(path):
    return joblib.load(path)
