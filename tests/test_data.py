import pandas as pd
import pytest

from smartstock.data import generate_synthetic_retail_data, time_based_split, validate_retail_data
from smartstock.features import build_model_frame


def test_validate_retail_data_rejects_duplicate_keys():
    df = generate_synthetic_retail_data(n_stores=1, n_items=1, days=40)
    broken = pd.concat([df, df.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError):
        validate_retail_data(broken)


def test_time_split_is_strictly_ordered():
    df = generate_synthetic_retail_data(n_stores=1, n_items=1, days=120)
    feature_frame = build_model_frame(df)
    train_df, validation_df, test_df, split = time_based_split(feature_frame)
    assert train_df["date"].max() <= split.train_end
    assert validation_df["date"].min() > split.train_end
    assert validation_df["date"].max() <= split.validation_end
    assert test_df["date"].min() > split.validation_end


def test_features_do_not_leak_future_values():
    df = generate_synthetic_retail_data(n_stores=1, n_items=1, days=60)
    feature_frame = build_model_frame(df, horizons=(7,))
    original = df.sort_values("date").reset_index(drop=True)
    second_day = feature_frame.iloc[1]
    assert second_day["lag_1"] == original.loc[0, "target_sales"]
    assert second_day["date"] == original.loc[1, "date"]
