from __future__ import annotations

import numpy as np
import pandas as pd


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def wape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denominator = np.abs(y_true).sum()
    if denominator == 0:
        return float(np.abs(y_true - y_pred).sum())
    return float(np.abs(y_true - y_pred).sum() / denominator)


def safe_mape(y_true: pd.Series, y_pred: pd.Series) -> float:
    denominator = np.where(np.abs(y_true) < 1e-6, np.nan, np.abs(y_true))
    return float(np.nanmean(np.abs((y_true - y_pred) / denominator)))


def summarize_metrics(evaluation_frame: pd.DataFrame, target_column: str, prediction_column: str) -> dict[str, float]:
    y_true = evaluation_frame[target_column]
    y_pred = evaluation_frame[prediction_column]
    return {
        "mae": round(mean_absolute_error(y_true, y_pred), 4),
        "rmse": round(root_mean_squared_error(y_true, y_pred), 4),
        "wape": round(wape(y_true, y_pred), 4),
        "safe_mape": round(safe_mape(y_true, y_pred), 4),
    }

