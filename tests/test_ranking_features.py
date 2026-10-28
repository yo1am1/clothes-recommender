from datetime import date

import numpy as np
import pandas as pd
import pytest

from scripts.ranking_features import FEATURE_NAMES, TemporalFeatures

AS_OF = date(2020, 1, 31)


@pytest.fixture
def frames():
    products = pd.DataFrame(
        {
            "product_id": [10, 20, 30, 40, 50],
            "product_code": [100, 100, 200, None, None],
            "colour": ["Red", " red ", "Blue", None, ""],
            "department_name": ["Tops", "TOPS", "Bottoms", None, pd.NA],
            "price": [9999] * 5,  # Full-window catalog prices must never enter features.
        }
    )
    purchases = pd.DataFrame(
        [
            (7, 10, "2020-01-01", 0.2, 1),
            (7, 10, "2020-01-25", 0.4, 2),
            (7, 30, "2020-01-31", 0.6, 2),
            (7, 40, "2020-01-24", 0.8, 1),
            (8, 10, "2020-02-01", 99.0, 2),
        ],
        columns=["user_id", "product_id", "purchase_date", "price_normalized", "sales_channel_id"],
    )
    crowd = pd.DataFrame(
        [
            (7, 10, "2020-01-31", 0.1, 2),
            (7, 10, "2020-01-25", 0.3, 1),
            (7, 10, "2020-01-24", 0.5, 2),
            (7, 10, "2020-01-04", 0.7, 1),
            (7, 10, "2020-01-03", 0.9, 2),
            (7, 10, "2020-01-01", 0.11, 2),
            (9, 20, "2020-01-04", 0.2, 2),
            (9, 30, "2020-01-31", None, None),
            (9, 40, "2018-06-01", 0.9, 1),
        ],
        columns=[
            "shopper_id",
            "product_id",
            "purchase_date",
            "price_normalized",
            "sales_channel_id",
        ],
    )
    return products, purchases, crowd


def named(row):
    return dict(zip(FEATURE_NAMES, row, strict=True))


def test_hand_calculated_features_and_original_price_units(frames):
    features = TemporalFeatures(*frames, AS_OF).features(7, [10, 20, 30, 40, 50])
    assert features.shape == (5, 18)
    row = named(features[0])
    assert row == pytest.approx(
        {
            "item_count_1d_log": np.log1p(1),
            "item_count_7d_log": np.log1p(2),
            "item_count_28d_log": np.log1p(4),
            "item_last_sale_days_log": 0,
            "item_first_sale_days_log": np.log1p(30),
            "item_trend_7d_28d": np.log1p(2) - np.log1p(4 / 4),
            "item_online_share_28d": 0.5,
            "item_price_median_28d": 0.4,
            "item_price_observed_28d": 1,
            "price_log_ratio": np.log(0.4 / 0.5),
            "user_price_missing": 0,
            "user_history_count_log": np.log1p(4),
            "user_recent_7d_count_log": np.log1p(2),
            "user_last_purchase_days_log": 0,
            "user_online_share": 0.5,
            "family_affinity": 0.5,
            "colour_affinity": 0.5,
            "department_affinity": 0.5,
        }
    )
    second = named(features[1])
    assert second["item_count_7d_log"] == 0
    assert second["item_count_28d_log"] == np.log1p(1)
    assert second["item_last_sale_days_log"] == np.log1p(27)
    assert second["price_log_ratio"] == pytest.approx(np.log(0.2 / 0.5))
    assert second["family_affinity"] == 0.5
    third = named(features[2])
    assert third["item_price_observed_28d"] == 0
    assert third["price_log_ratio"] == 0
    assert third["family_affinity"] == 0.25
    for row in map(named, features[3:]):
        assert row["item_last_sale_days_log"] == np.log1p(365)
        assert row["item_first_sale_days_log"] == np.log1p(365)
        assert row["item_price_median_28d"] == 0
        assert row["family_affinity"] == row["colour_affinity"] == row["department_affinity"] == 0
    assert np.isfinite(features).all()


def test_future_events_and_catalog_prices_cannot_change_features(frames):
    products, purchases, crowd = frames
    expected = TemporalFeatures(*frames, AS_OF).features(7, [10, 20, 30, 40, 50])
    future_personal = purchases.iloc[:1].assign(
        purchase_date="2020-02-01", product_id=50, price_normalized=5000, sales_channel_id=2
    )
    future_crowd = crowd.iloc[:1].assign(
        purchase_date="2020-02-01", product_id=20, price_normalized=1e20, sales_channel_id=1
    )
    # Duplicate DataFrame indices are legitimate after concatenating event shards.
    actual = TemporalFeatures(
        products.assign(price=1e30),
        pd.concat([purchases, future_personal]),
        pd.concat([crowd, future_crowd]),
        AS_OF,
    ).features(7, [10, 20, 30, 40, 50])
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("user", [8, 999])
def test_no_past_history_has_finite_defaults(frames, user):
    row = named(TemporalFeatures(*frames, AS_OF).features(user, [10])[0])
    assert row["price_log_ratio"] == 0
    assert row["user_price_missing"] == 1
    assert row["user_history_count_log"] == row["user_recent_7d_count_log"] == 0
    assert row["user_last_purchase_days_log"] == np.log1p(365)
    assert row["user_online_share"] == 0
    assert row["family_affinity"] == row["colour_affinity"] == row["department_affinity"] == 0
    assert np.isfinite(list(row.values())).all()


def test_missing_metadata_is_not_a_matching_category(frames):
    products, purchases, crowd = frames
    products = products.assign(product_code=None, colour=" ", department_name=pd.NA)
    actual = TemporalFeatures(products, purchases, crowd, AS_OF).features(7, [10, 20, 40, 50])
    np.testing.assert_array_equal(actual[:, -3:], np.zeros((4, 3)))


def test_alignment_empty_rows_unknown_ids_and_input_snapshots(frames):
    originals = [frame.copy(deep=True) for frame in frames]
    model = TemporalFeatures(*frames, AS_OF)
    rows = model.features(7, [20, 10, 20])
    np.testing.assert_array_equal(rows[0], rows[2])
    assert named(rows[0])["item_count_1d_log"] == 0
    assert named(rows[1])["item_count_1d_log"] == np.log1p(1)
    assert model.features(7, []).shape == (0, len(FEATURE_NAMES))
    with pytest.raises(ValueError, match="catalog"):
        model.features(7, [999])
    for original, frame in zip(originals, frames, strict=True):
        pd.testing.assert_frame_equal(original, frame)
    frames[2].loc[:, "price_normalized"] = 999
    np.testing.assert_array_equal(model.features(7, [20, 10, 20]), rows)


@pytest.mark.parametrize(
    "which,missing",
    [
        (which, column)
        for which, identity in [(1, "user_id"), (2, "shopper_id")]
        for column in [
            identity,
            "product_id",
            "purchase_date",
            "price_normalized",
            "sales_channel_id",
        ]
    ]
    + [(0, column) for column in ["product_id", "product_code", "colour", "department_name"]],
)
def test_enriched_schema_is_required(frames, which, missing):
    frames = list(frames)
    frames[which] = frames[which].drop(columns=missing)
    with pytest.raises(ValueError, match=f"missing required columns.*{missing}"):
        TemporalFeatures(*frames, AS_OF)


def test_empty_events_and_missing_prices_are_safe(frames):
    products, purchases, crowd = frames
    empty = TemporalFeatures(products, purchases.iloc[:0], crowd.iloc[:0], AS_OF).features(7, [10])
    assert np.isfinite(empty).all()
    assert named(empty[0])["item_price_observed_28d"] == 0
    missing = TemporalFeatures(
        products, purchases.assign(price_normalized=np.inf), crowd.assign(price_normalized=0), AS_OF
    ).features(7, [10])
    row = named(missing[0])
    assert row["item_price_observed_28d"] == 0
    assert row["user_price_missing"] == 1
    assert row["price_log_ratio"] == 0
    assert np.isfinite(missing).all()
