import pandas as pd

from smartstock.inventory import build_inventory_recommendations


def test_reorder_quantity_never_negative():
    forecast_frame = pd.DataFrame(
        {
            "forecast_date": pd.to_datetime(["2024-01-08"]),
            "store_id": ["store_1"],
            "item_id": ["item_01"],
            "category": ["Produce"],
            "predicted_demand": [10.0],
            "forecast_horizon": [7],
            "recent_mean": [5.0],
            "recent_std": [1.5],
        }
    )
    result = build_inventory_recommendations(forecast_frame, current_inventory_multiplier=10.0)
    assert result["recommended_reorder_qty"].iloc[0] == 0.0


def test_higher_forecast_increases_reorder_quantity():
    forecast_frame = pd.DataFrame(
        {
            "forecast_date": pd.to_datetime(["2024-01-08", "2024-01-08"]),
            "store_id": ["store_1", "store_1"],
            "item_id": ["item_01", "item_02"],
            "category": ["Produce", "Produce"],
            "predicted_demand": [20.0, 40.0],
            "forecast_horizon": [7, 7],
            "recent_mean": [3.0, 3.0],
            "recent_std": [1.0, 1.0],
        }
    )
    result = build_inventory_recommendations(forecast_frame, current_inventory_multiplier=0.8)
    assert result["recommended_reorder_qty"].max() > result["recommended_reorder_qty"].min()


def test_low_inventory_and_high_demand_raise_stockout_risk():
    forecast_frame = pd.DataFrame(
        {
            "forecast_date": pd.to_datetime(["2024-01-08"]),
            "store_id": ["store_1"],
            "item_id": ["item_01"],
            "category": ["Produce"],
            "predicted_demand": [100.0],
            "forecast_horizon": [7],
            "recent_mean": [2.0],
            "recent_std": [5.0],
        }
    )
    result = build_inventory_recommendations(forecast_frame, current_inventory_multiplier=0.4)
    assert result["stockout_risk"].iloc[0] == "High"

