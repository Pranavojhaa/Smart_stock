from __future__ import annotations

import json

import pandas as pd
import plotly.express as px
import streamlit as st

from smartstock.config import FORECASTS_PATH, METRICS_PATH, PROCESSED_DATA_PATH, RECOMMENDATIONS_PATH
from smartstock.inventory import build_inventory_recommendations
from smartstock.pipeline import bootstrap_demo_artifacts


st.set_page_config(
    page_title="SmartStock",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if not (PROCESSED_DATA_PATH.exists() and FORECASTS_PATH.exists() and RECOMMENDATIONS_PATH.exists() and METRICS_PATH.exists()):
        bootstrap_demo_artifacts(force=True)

    feature_frame = pd.read_csv(PROCESSED_DATA_PATH, parse_dates=["date"])
    forecasts = pd.read_csv(FORECASTS_PATH, parse_dates=["forecast_date"])
    with METRICS_PATH.open("r", encoding="utf-8") as file_obj:
        metrics = json.load(file_obj)
    return feature_frame, forecasts, metrics


def apply_simulator(
    forecasts: pd.DataFrame,
    lead_time_days: int,
    service_level_factor: float,
    current_inventory_multiplier: float,
) -> pd.DataFrame:
    return build_inventory_recommendations(
        forecasts,
        lead_time_days=lead_time_days,
        service_level_factor=service_level_factor,
        current_inventory_multiplier=current_inventory_multiplier,
    )


feature_frame, forecasts, metrics = load_artifacts()

st.title("SmartStock")
st.caption("Demand forecasting and inventory intelligence for retail planners.")

with st.sidebar:
    st.header("Inventory Simulator")
    selected_horizon = st.selectbox("Forecast horizon", options=[7, 28], index=0)
    lead_time_days = st.slider("Lead time (days)", min_value=3, max_value=28, value=7, step=1)
    service_level_factor = st.slider("Service buffer (z-score)", min_value=0.5, max_value=2.5, value=1.65, step=0.05)
    current_inventory_multiplier = st.slider("Current inventory coverage", min_value=0.4, max_value=1.6, value=0.9, step=0.05)

filtered_forecasts = forecasts[forecasts["forecast_horizon"] == selected_horizon].copy()
recommendations = apply_simulator(
    filtered_forecasts,
    lead_time_days=lead_time_days,
    service_level_factor=service_level_factor,
    current_inventory_multiplier=current_inventory_multiplier,
)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Forecast rows", f"{len(recommendations):,}")
col2.metric("Avg reorder qty", f"{recommendations['recommended_reorder_qty'].mean():.1f}")
col3.metric("High risk items", int((recommendations["stockout_risk"] == "High").sum()))
col4.metric("Overstock flags", int((recommendations["inventory_status"] == "Overstock risk").sum()))

st.subheader("Business Context")
st.markdown(
    """
SmartStock is designed for three portfolio-ready personas:

- Inventory managers who need reorder guidance by store and item.
- Category planners comparing demand shifts across categories and time horizons.
- Operations analysts tracking stockout risk, excess inventory, and forecast quality.
"""
)

tab_overview, tab_forecast, tab_inventory, tab_errors = st.tabs(
    ["Overview", "Forecast Explorer", "Inventory Planner", "Model Diagnostics"]
)

with tab_overview:
    history = feature_frame.groupby("date", as_index=False)["target_sales"].sum()
    history_chart = px.line(history, x="date", y="target_sales", title="Daily portfolio demand")
    st.plotly_chart(history_chart, use_container_width=True)

    category_chart = px.box(
        feature_frame,
        x="category",
        y="target_sales",
        color="category",
        title="Demand distribution by category",
    )
    st.plotly_chart(category_chart, use_container_width=True)

with tab_forecast:
    store_options = sorted(recommendations["store_id"].unique())
    category_options = sorted(recommendations["category"].unique())
    selected_store = st.selectbox("Store", options=store_options, index=0)
    selected_category = st.selectbox("Category", options=category_options, index=0)
    scoped = recommendations[
        (recommendations["store_id"] == selected_store) & (recommendations["category"] == selected_category)
    ].copy()

    st.dataframe(
        scoped[
            [
                "item_id",
                "predicted_horizon_demand",
                "current_inventory",
                "recommended_reorder_qty",
                "stockout_risk",
                "inventory_status",
            ]
        ],
        use_container_width=True,
        hide_index=True,
    )
    demand_bar = px.bar(
        scoped.sort_values("predicted_horizon_demand", ascending=False),
        x="item_id",
        y="predicted_horizon_demand",
        color="stockout_risk",
        title="Predicted demand by item",
    )
    st.plotly_chart(demand_bar, use_container_width=True)

with tab_inventory:
    top_risk = recommendations.sort_values(["stockout_risk", "recommended_reorder_qty"], ascending=[True, False]).head(15)
    reorder_chart = px.bar(
        top_risk.sort_values("recommended_reorder_qty", ascending=True),
        x="recommended_reorder_qty",
        y="item_id",
        color="inventory_status",
        orientation="h",
        title="Top reorder actions",
        hover_data=["store_id", "category", "predicted_horizon_demand", "current_inventory"],
    )
    st.plotly_chart(reorder_chart, use_container_width=True)
    st.dataframe(recommendations.head(30), use_container_width=True, hide_index=True)

with tab_errors:
    model_metrics = pd.DataFrame.from_dict(metrics["model_metrics"], orient="index")
    baseline_metrics = pd.DataFrame.from_dict(metrics["baseline_metrics"], orient="index")
    metric_table = pd.DataFrame(
        {
            "horizon": model_metrics["horizon"].astype(int),
            "model_mae": model_metrics["metrics"].apply(lambda row: row["mae"]),
            "model_rmse": model_metrics["metrics"].apply(lambda row: row["rmse"]),
            "model_wape": model_metrics["metrics"].apply(lambda row: row["wape"]),
            "baseline_wape": baseline_metrics["wape"].astype(float),
        }
    ).sort_values("horizon")
    st.dataframe(metric_table, use_container_width=True, hide_index=True)

    drift = feature_frame.groupby(["date", "category"], as_index=False)["target_sales"].mean()
    drift_chart = px.line(drift, x="date", y="target_sales", color="category", title="Category demand drift over time")
    st.plotly_chart(drift_chart, use_container_width=True)

st.caption("Tip: run `python -m smartstock.pipeline` once to regenerate artifacts, or let the app bootstrap them automatically.")

