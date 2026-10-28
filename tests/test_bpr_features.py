import sys
from datetime import date
from types import ModuleType, SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import isspmatrix_csr

from scripts.bpr_features import FEATURE_NAMES, BPRFeatures

AS_OF = date(2020, 1, 7)


@pytest.fixture
def frames():
    products = pd.DataFrame({"product_id": [30, 10, 40, 20]})
    users = pd.DataFrame({"user_id": [1, 9, 2, 3], "shopper_id": [99, 1, 77, 66]})
    crowd = pd.DataFrame(
        [
            (99, 30, "2020-01-07"),
            (99, 10, "2020-01-01"),
            (99, 10, "2020-01-02"),
            (1, 20, "2020-01-07"),
            (1, 10, "2020-01-05"),
            (77, 40, "2020-01-08"),
            (66, 999, "2020-01-04"),
        ],
        columns=["shopper_id", "product_id", "purchase_date"],
    )
    return products, users, crowd.assign(sales_channel_id=2, price_normalized=0.1)


@pytest.fixture
def fake_bpr(monkeypatch):
    state = SimpleNamespace(params=[], matrices=[], user=None, item=None)

    class Model:
        def __init__(self, **kwargs):
            state.params.append(kwargs)

        def fit(self, matrix, *, show_progress):
            assert show_progress is False
            assert isspmatrix_csr(matrix)
            state.matrices.append(matrix.copy())
            # The final coordinate models an item bias, not a normalized similarity.
            self.user_factors = (
                np.array([[1.0, 1.0], [2.0, 1.0]], dtype=np.float32)
                if state.user is None
                else state.user
            )
            self.item_factors = (
                np.array([[1.0, 0.5], [2.0, -0.5], [2.0, -0.5]], dtype=np.float32)
                if state.item is None
                else state.item
            )

    for name in ["implicit", "implicit.cpu", "implicit.cpu.bpr"]:
        module = ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(
        sys.modules["implicit.cpu.bpr"], "BayesianPersonalizedRanking", Model, raising=False
    )
    return state


def test_sorted_binary_matrix_crosswalk_bias_ties_and_duplicate_alignment(frames, fake_bpr):
    model = BPRFeatures(*frames, AS_OF)
    assert FEATURE_NAMES == ("bpr_score", "bpr_candidate_reciprocal_rank", "bpr_known_vector")
    assert fake_bpr.params == [
        {
            "factors": 64,
            "iterations": 50,
            "learning_rate": 0.05,
            "regularization": 0.01,
            "random_state": 42,
            "num_threads": 1,
            "verify_negative_samples": True,
        }
    ]
    matrix = fake_bpr.matrices[0]
    assert matrix.dtype == np.float32 and matrix.has_sorted_indices
    # Sorted shoppers [1,99], sorted observed catalog items [10,20,30].
    np.testing.assert_array_equal(matrix.toarray(), [[1, 1, 0], [1, 0, 1]])
    actual = model.features(1, [30, 10, 20, 20, 40])
    np.testing.assert_allclose(
        actual, [[3.5, 0.5, 1], [2.5, 1 / 3, 1], [3.5, 1, 1], [3.5, 1, 1], [0, 0, 0]]
    )
    assert actual.dtype == np.float64
    # user1 belongs to shopper99, despite a different crowd shopper with integer ID1.
    assert model.features(9, [10])[0, 0] == 1.5
    assert actual[1, 0] == 2.5  # An owned/repeat item remains scoreable.
    assert model.features(1, [30])[0, 1] == 1  # Rank is within supplied unique known IDs.


def test_future_rows_event_weights_channels_and_input_order_do_not_change_features(
    frames, fake_bpr
):
    products, users, crowd = frames
    first = BPRFeatures(*frames, AS_OF).features(1, [10, 20, 30, 40])
    future = crowd.iloc[:2].assign(purchase_date="2020-01-08", shopper_id=77, product_id=40)
    changed = pd.concat([crowd, future]).iloc[::-1].assign(price_normalized=999, sales_channel_id=1)
    second = BPRFeatures(products.iloc[::-1], users.iloc[::-1], changed, AS_OF).features(
        1, [10, 20, 30, 40]
    )
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(fake_bpr.matrices[0].toarray(), fake_bpr.matrices[1].toarray())


def test_unknown_user_or_missing_past_vectors_get_zero_defaults(frames, fake_bpr):
    model = BPRFeatures(*frames, AS_OF)
    for user_id in [2, 3, 999]:
        np.testing.assert_array_equal(model.features(user_id, [10, 40]), np.zeros((2, 3)))
    assert model.features(1, []).shape == (0, 3)
    with pytest.raises(ValueError, match="catalog"):
        model.features(999, [999])


def test_raw_negative_scores_are_not_clipped(frames, fake_bpr):
    fake_bpr.user = np.array([[1.0, 1.0], [-1.0, 1.0]])
    actual = BPRFeatures(*frames, AS_OF).features(1, [20, 10])
    np.testing.assert_allclose(actual, [[-2.5, 0.5, 1], [-0.5, 1, 1]])


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [(99, 10)],
        [(99, 10), (1, 10)],
        [(99, 10), (99, 20)],
        [(99, 10), (99, 20), (1, 10), (1, 20)],
    ],
)
def test_empty_tiny_or_no_negative_matrices_are_not_fitted(frames, fake_bpr, rows):
    products, users, _ = frames
    crowd = pd.DataFrame(rows, columns=["shopper_id", "product_id"]).assign(
        purchase_date="2020-01-07"
    )
    model = BPRFeatures(products, users, crowd, AS_OF)
    assert not fake_bpr.params and not fake_bpr.matrices
    np.testing.assert_array_equal(model.features(1, [10, 20]), np.zeros((2, 3)))


@pytest.mark.parametrize(
    "field,value",
    [
        ("user", np.ones((1, 2))),
        ("user", np.ones(2)),
        ("user", np.empty((2, 0))),
        ("item", np.ones((3, 3))),
        ("item", np.ones((2, 2))),
        ("user", np.full((2, 2), np.nan)),
        ("item", np.full((3, 2), np.inf)),
        ("item", np.full((3, 2), "invalid")),
    ],
)
def test_invalid_factor_arrays_fail_closed(frames, fake_bpr, field, value):
    setattr(fake_bpr, field, value)
    with pytest.raises(ValueError, match="aligned finite"):
        BPRFeatures(*frames, AS_OF)


@pytest.mark.parametrize(
    "column,values",
    [
        ("user_id", [1, 1, 2, 3]),
        ("shopper_id", [99, 99, 77, 66]),
        ("user_id", [1, None, 2, 3]),
        ("shopper_id", [99, None, 77, 66]),
    ],
)
def test_crosswalk_must_be_bijective_and_nonmissing(frames, fake_bpr, column, values):
    products, users, crowd = frames
    with pytest.raises(ValueError, match=r"crosswalk.*unique and nonmissing"):
        BPRFeatures(products, users.assign(**{column: values}), crowd, AS_OF)
    assert not fake_bpr.params


@pytest.mark.parametrize(
    "which,column",
    [
        (0, "product_id"),
        (1, "user_id"),
        (1, "shopper_id"),
        (2, "shopper_id"),
        (2, "product_id"),
        (2, "purchase_date"),
    ],
)
def test_required_columns_fail_clearly(frames, fake_bpr, which, column):
    frames = list(frames)
    frames[which] = frames[which].drop(columns=column)
    with pytest.raises(ValueError, match=f"missing required columns.*{column}"):
        BPRFeatures(*frames, AS_OF)


def test_missing_optional_dependency_fails_before_fitting(frames, fake_bpr, monkeypatch):
    monkeypatch.setitem(sys.modules, "implicit.cpu.bpr", None)
    with pytest.raises(ValueError, match="optional CPU implicit"):
        BPRFeatures(*frames, AS_OF)
    assert not fake_bpr.params
