from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import issparse

from clothes_recommender.catalog import CustomerData
from clothes_recommender.recommenders.base import UnknownUserError
from scripts import retrieval_channels
from scripts.retrieval_channels import CHANNELS, ComplementaryRetrieval

DAY = date(2020, 9, 1)
YESTERDAY = DAY - timedelta(days=1)
TOMORROW = DAY + timedelta(days=1)


def customers(rows=(), user_ids=(1, 2, 3, 4)):
    return CustomerData(
        pd.DataFrame({"user_id": user_ids, "age": [30] * len(user_ids)}),
        pd.DataFrame(rows, columns=["user_id", "product_id", "purchase_date"]),
    )


def crowd(rows=()):
    return pd.DataFrame(rows, columns=["shopper_id", "product_id", "purchase_date"])


@pytest.fixture
def products():
    return pd.DataFrame(
        {
            "product_id": list(range(10, 20)),
            "product_code": [700, 700, 701, 800, 800, None, None, 900, 900, 700],
            "name": ["Same misleading name"] * 10,
        }
    )


@pytest.fixture
def purchases():
    return [
        (100, 10, DAY),
        (100, 11, DAY),
        (100, 12, DAY),
        (101, 10, YESTERDAY),
        (101, 11, YESTERDAY),
        (101, 11, YESTERDAY),
        (102, 13, DAY - timedelta(days=2)),
        (102, 14, DAY - timedelta(days=2)),
        (103, 13, DAY - timedelta(days=3)),
        (103, 14, DAY - timedelta(days=3)),
        (104, 12, DAY),
        (104, 15, DAY),
        (105, 16, DAY),
        (106, 17, DAY - timedelta(days=40)),
        (106, 18, DAY - timedelta(days=40)),
        (107, 17, DAY - timedelta(days=39)),
        (107, 18, DAY - timedelta(days=39)),
    ]


def test_hand_checked_family_and_basket_support(products, purchases):
    model = ComplementaryRetrieval(
        products,
        customers([(1, 10, DAY), (1, 13, DAY - timedelta(days=7))]),
        crowd(purchases),
        DAY,
    )
    scores = model.scores(1, [10, 11, 12, 13, 14, 19, 999, 11])
    # Same actual product_code only; a repeated row is not another popularity basket.
    np.testing.assert_allclose(
        scores["family"], [0, 1 + np.log(3), 0, 0, 0.5 * (1 + np.log(3)), 0, 0, 1 + np.log(3)]
    )
    # Six recent baskets; each supported pair occurs twice, with both margins equal to two.
    np.testing.assert_allclose(
        scores["recent_basket"], [0, np.log(4) / 3, 0, 0, np.log(4) / 6, 0, 0, np.log(4) / 3]
    )
    assert [p.product_id for p in model.channels(1)["family"]] == [11, 14]
    assert [p.product_id for p in model.channels(1)["recent_basket"]] == [11, 14]


def test_future_crowd_and_personal_rows_do_not_change_any_channel(products, purchases):
    past = [(1, 10, DAY), (1, 13, DAY - timedelta(days=7))]
    before = ComplementaryRetrieval(products, customers(past), crowd(purchases), DAY)
    after = ComplementaryRetrieval(
        products,
        customers([*past, (1, 17, TOMORROW), (1, 11, TOMORROW), (4, 19, TOMORROW)]),
        crowd(
            [
                *purchases,
                (999, 19, TOMORROW),
                (999, 10, TOMORROW),
                (998, 19, TOMORROW),
                (998, 10, TOMORROW),
            ]
        ),
        DAY,
    )
    for name in CHANNELS:
        np.testing.assert_allclose(
            before.scores(1, range(10, 20))[name], after.scores(1, range(10, 20))[name], atol=1e-12
        )
        assert after.scores(1, [19])[name][0] == 0
        assert after.channels(4)[name] == []


def test_duplicate_rows_and_old_baskets_do_not_inflate_support(products, purchases):
    history = customers([(1, 10, DAY), (1, 17, DAY - timedelta(days=40))])
    original = ComplementaryRetrieval(products, history, crowd(purchases), DAY)
    duplicated = ComplementaryRetrieval(products, history, crowd(purchases * 3), DAY)
    for name in CHANNELS:
        np.testing.assert_allclose(
            original.scores(1, range(10, 20))[name],
            duplicated.scores(1, range(10, 20))[name],
            atol=1e-12,
        )
    assert original.scores(1, [18])["recent_basket"][0] == 0
    assert original.scores(1, [18])["family"][0] == 0


def test_distinct_days_of_same_shopper_are_distinct_baskets(products):
    rows = [(100, pid, day) for day in (DAY, YESTERDAY) for pid in (10, 11)]
    model = ComplementaryRetrieval(products, customers([(1, 10, DAY)]), crowd(rows), DAY)
    assert model.scores(1, [11])["recent_basket"][0] == pytest.approx(np.log(2))


def test_latent_folds_personal_history_not_numerically_colliding_crowd_id(products, purchases):
    # User 100 collides with a crowd shopper who bought 10,11,12; user 999 does not.
    # Both personal histories are identical and must fold into identical item profiles.
    history = customers([(100, 13, DAY), (999, 13, DAY)], user_ids=(100, 999))
    model = ComplementaryRetrieval(products, history, crowd(purchases), DAY)
    left, right = model.scores(100, range(10, 20)), model.scores(999, range(10, 20))
    for name in CHANNELS:
        np.testing.assert_allclose(left[name], right[name])
    assert left["latent"][4] > 0  # product 14 is learned from personal seed 13
    assert left["latent"][3] == 0  # exact owned product 13 stays excluded
    assert left["latent"][7] == 0  # disconnected 17 is not numerical-noise evidence


def test_svd_is_sparse_seeded_and_rank_bounded(monkeypatch):
    actual = retrieval_channels.svds
    calls = []

    def checked(matrix, **kwargs):
        assert issparse(matrix)
        assert kwargs == {"k": 32, "return_singular_vectors": "vh", "rng": 42}
        calls.append(matrix.shape)
        return actual(matrix, **kwargs)

    monkeypatch.setattr(retrieval_channels, "svds", checked)
    products = pd.DataFrame({"product_id": range(40)})
    rows = [
        (shopper, pid, DAY) for shopper in range(45) for pid in (shopper % 40, (shopper + 1) % 40)
    ]
    model = ComplementaryRetrieval(products, customers([(1, 0, DAY)]), crowd(rows), DAY)
    assert calls == [(45, 40)]
    assert model._factors.shape == (40, 32)
    assert all(np.isfinite(scores).all() for scores in model.scores(1, range(40)).values())


def test_finite_deterministic_outputs_and_exclusions(products, purchases):
    history = customers([(1, 10, DAY), (1, 13, YESTERDAY)])
    first = ComplementaryRetrieval(products, history, crowd(purchases), DAY)
    second = ComplementaryRetrieval(products, history, crowd(purchases), DAY)
    for name in CHANNELS:
        np.testing.assert_allclose(
            first.scores(1, range(10, 20))[name], second.scores(1, range(10, 20))[name], atol=1e-12
        )
        results = first.channels(1, top_k=2)[name]
        assert len(results) <= 2
        assert all(
            p.product_id not in {10, 13, 19} and np.isfinite(p.score) and p.score > 0
            for p in results
        )
    assert first.channels(1, 0) == {name: [] for name in CHANNELS}
    assert first.channels(1, -1) == {name: [] for name in CHANNELS}


def test_family_ties_by_id_and_requires_actual_nonempty_codes():
    products = pd.DataFrame({"product_id": [12, 10, 11], "product_code": [700, 700, 700]})
    history = customers([(1, 10, DAY), (1, 10, YESTERDAY)])
    rows = crowd([(100, 11, DAY), (101, 12, DAY)])
    model = ComplementaryRetrieval(products, history, rows, DAY)
    assert [p.product_id for p in model.channels(1)["family"]] == [11, 12]
    missing = ComplementaryRetrieval(products.drop(columns="product_code"), history, rows, DAY)
    blank = ComplementaryRetrieval(
        products.assign(product_code=[None, "", None]), history, rows, DAY
    )
    assert missing.channels(1)["family"] == blank.channels(1)["family"] == []


@pytest.mark.parametrize("rows", [[], [(100, 10, DAY)], [(100, 10, DAY), (100, 11, DAY)]])
def test_empty_small_and_no_history_do_not_fill(products, rows):
    history = customers([(1, 10, DAY), (4, 13, TOMORROW)])
    model = ComplementaryRetrieval(products, history, crowd(rows), DAY)
    assert model.channels(2) == {name: [] for name in CHANNELS}
    assert model.channels(4) == {name: [] for name in CHANNELS}
    assert model.channels(1)["latent"] == []
    for values in model.scores(1, []).values():
        assert values.shape == (0,)
    assert all(np.isfinite(values).all() for values in model.scores(1, [10, 999]).values())
    empty_catalog = ComplementaryRetrieval(products.iloc[:0], history, crowd(rows), DAY)
    assert empty_catalog.channels(1) == {name: [] for name in CHANNELS}


def test_unknown_user_raises_even_for_empty_request(products):
    model = ComplementaryRetrieval(products, customers(), crowd(), DAY)
    with pytest.raises(UnknownUserError):
        model.channels(999, top_k=0)
    with pytest.raises(UnknownUserError):
        model.scores(999, [])


def test_recent_window_includes_cutoff_minus_27_but_not_minus_28(products):
    rows = crowd(
        [
            (shopper, pid, DAY - timedelta(days=age))
            for shopper, age in [(100, 27), (101, 28)]
            for pid in (10, 11)
        ]
    )
    history = customers([(1, 10, DAY - timedelta(days=27)), (3, 10, DAY - timedelta(days=28))])
    model = ComplementaryRetrieval(products, history, rows, DAY)
    assert model.scores(1, [11])["family"][0] > 0
    assert model.scores(3, [11])["family"][0] == 0
    assert model.scores(1, [11])["recent_basket"][0] == 0  # only one in-window basket


def test_very_old_crowd_and_history_remain_finite(products):
    old = date(1900, 1, 1)
    rows = crowd([(shopper, pid, old) for shopper in (100, 101, 102) for pid in (10, 11)])
    model = ComplementaryRetrieval(products, customers([(1, 10, old)]), rows, DAY)
    assert all(np.isfinite(values).all() for values in model.scores(1, [10, 11]).values())
    assert model.channels(1) == {name: [] for name in CHANNELS}
