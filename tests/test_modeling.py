from smartstock.baseline import add_baseline_predictions
from smartstock.data import generate_synthetic_retail_data, time_based_split
from smartstock.features import build_latest_snapshot, build_model_frame, get_feature_columns
from smartstock.metrics import safe_mape, summarize_metrics
from smartstock.modeling import train_models


def test_baseline_runs_on_full_supported_dataset():
    df = generate_synthetic_retail_data(n_stores=2, n_items=3, days=120)
    frame = build_model_frame(df, horizons=(7,))
    scored = add_baseline_predictions(frame, horizon=7)
    assert "baseline_seasonal_h7" in scored.columns
    assert scored["baseline_seasonal_h7"].notna().sum() > 0


def test_primary_model_trains_and_predicts_requested_horizons():
    df = generate_synthetic_retail_data(n_stores=2, n_items=3, days=180)
    frame = build_model_frame(df, horizons=(7, 28))
    train_df, validation_df, _, split = time_based_split(frame)
    feature_columns = get_feature_columns(frame)
    models, metrics, _ = train_models(train_df, validation_df, feature_columns, horizons=(7, 28))
    snapshot = build_latest_snapshot(frame, split.validation_end)
    preds_7 = models[7].predict(snapshot[feature_columns])
    preds_28 = models[28].predict(snapshot[feature_columns])
    assert len(preds_7) == len(snapshot)
    assert len(preds_28) == len(snapshot)
    assert metrics[7].metrics["wape"] >= 0


def test_metrics_handle_zero_demand_periods():
    assert safe_mape(
        y_true=__import__("pandas").Series([0.0, 10.0, 0.0, 5.0]),
        y_pred=__import__("pandas").Series([1.0, 8.0, 0.0, 4.5]),
    ) >= 0

    summary = summarize_metrics(
        __import__("pandas").DataFrame({"actual": [0.0, 0.0, 5.0], "pred": [0.0, 1.0, 4.0]}),
        "actual",
        "pred",
    )
    assert summary["wape"] >= 0

