from __future__ import annotations

import numpy as np
import pandas as pd


def compute_safety_stock(recent_std: pd.Series, lead_time_days: int, service_level_factor: float = 1.65) -> pd.Series:
    return service_level_factor * recent_std.fillna(0.0) * np.sqrt(max(1, lead_time_days))


def classify_inventory_status(coverage_ratio: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [coverage_ratio < 0.9, coverage_ratio > 1.35],
            ["Stockout risk", "Overstock risk"],
            default="Healthy",
        ),
        index=coverage_ratio.index,
    )


def score_stockout_risk(coverage_ratio: pd.Series) -> pd.Series:
    return pd.Series(
        np.select(
            [coverage_ratio < 0.75, coverage_ratio < 1.0, coverage_ratio < 1.2],
            ["High", "Medium", "Low"],
            default="Low",
        ),
        index=coverage_ratio.index,
    )


def build_inventory_recommendations(
    forecast_frame: pd.DataFrame,
    lead_time_days: int = 7,
    service_level_factor: float = 1.65,
    current_inventory_multiplier: float = 0.9,
) -> pd.DataFrame:
    recommendations = forecast_frame.copy()
    recommendations["current_inventory"] = (
        recommendations["recent_mean"].fillna(recommendations["predicted_demand"] / recommendations["forecast_horizon"]).fillna(0.0)
        * recommendations["forecast_horizon"]
        * current_inventory_multiplier
    ).round(2)
    recommendations["safety_stock"] = compute_safety_stock(
        recommendations["recent_std"], lead_time_days, service_level_factor
    ).round(2)
    recommendations["predicted_horizon_demand"] = recommendations["predicted_demand"].round(2)
    recommendations["recommended_reorder_qty"] = (
        recommendations["predicted_horizon_demand"] + recommendations["safety_stock"] - recommendations["current_inventory"]
    ).clip(lower=0).round(2)
    recommendations["coverage_ratio"] = np.where(
        recommendations["predicted_horizon_demand"] <= 0,
        2.0,
        recommendations["current_inventory"] / recommendations["predicted_horizon_demand"],
    )
    recommendations["stockout_risk"] = score_stockout_risk(recommendations["coverage_ratio"])
    recommendations["inventory_status"] = classify_inventory_status(recommendations["coverage_ratio"])
    recommendations["estimated_excess_inventory"] = (
        recommendations["current_inventory"] - recommendations["predicted_horizon_demand"]
    ).clip(lower=0).round(2)
    recommendations["estimated_stockout_units"] = (
        recommendations["predicted_horizon_demand"] - recommendations["current_inventory"]
    ).clip(lower=0).round(2)
    recommendations["estimated_stockout_days"] = np.where(
        recommendations["predicted_horizon_demand"] <= 0,
        0,
        (recommendations["estimated_stockout_units"] / (recommendations["predicted_horizon_demand"] / recommendations["forecast_horizon"])).round(1),
    )

    return recommendations[
        [
            "forecast_date",
            "store_id",
            "item_id",
            "category",
            "forecast_horizon",
            "predicted_horizon_demand",
            "current_inventory",
            "safety_stock",
            "recommended_reorder_qty",
            "stockout_risk",
            "inventory_status",
            "estimated_excess_inventory",
            "estimated_stockout_units",
            "estimated_stockout_days",
            "recent_mean",
            "recent_std",
        ]
    ].sort_values(["forecast_horizon", "recommended_reorder_qty"], ascending=[True, False])

