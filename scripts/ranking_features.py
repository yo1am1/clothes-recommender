"""Past-only experimental ranking features; no serving model or artifact writes.

Counts are purchase-event counts, transformed with log1p. Calendar windows include
as_of and the preceding N-1 days. Age features are log1p(days), capped at 365;
unobserved ages use 365. Trend is log1p(count7)-log1p(count28/4).
Prices remain in original normalized transaction units.
Online shares average known channels (1/2); missing shares default to zero.
Affinities count matching events / all personal events, using trimmed, case-folded
metadata; missing/blank values never match. No full-window catalog price is used.
"""

from collections import Counter
from collections.abc import Iterable
from datetime import date

import numpy as np
import pandas as pd

FEATURE_NAMES = (
    "item_count_1d_log",
    "item_count_7d_log",
    "item_count_28d_log",
    "item_last_sale_days_log",
    "item_first_sale_days_log",
    "item_trend_7d_28d",
    "item_online_share_28d",
    "item_price_median_28d",
    "item_price_observed_28d",
    "price_log_ratio",
    "user_price_missing",
    "user_history_count_log",
    "user_recent_7d_count_log",
    "user_last_purchase_days_log",
    "user_online_share",
    "family_affinity",
    "colour_affinity",
    "department_affinity",
)
METADATA = ("product_code", "colour", "department_name")


def _require(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{name} missing required columns: {sorted(missing)}")


def _past(frame: pd.DataFrame, identity: str, as_of: pd.Timestamp) -> pd.DataFrame:
    columns = (identity, "product_id", "purchase_date", "price_normalized", "sales_channel_id")
    _require(frame, columns, identity)
    try:
        dates = pd.to_datetime(frame["purchase_date"]).dt.normalize()
        if dates.isna().any():
            raise ValueError("missing purchase_date")
        keep = dates <= as_of
        past = frame.loc[keep, list(columns)].copy()
    except (ValueError, TypeError) as error:
        raise ValueError("Events need valid timezone-naive purchase_date values") from error
    if past[[identity, "product_id"]].isna().any().any():
        raise ValueError(f"Past events need {identity} and product_id")
    past["purchase_date"] = dates[keep].to_numpy()
    price = pd.to_numeric(past["price_normalized"], errors="coerce")
    past["price_normalized"] = price.where(np.isfinite(price) & (price > 0))
    past["online"] = past["sales_channel_id"].map({1: 0.0, 2: 1.0})
    return past


class TemporalFeatures:
    """Preaggregate crowd/item and personal statistics once, then look up aligned rows.

    Item median/observed flag use positive finite event prices within 28 days. The
    user median uses all past positive finite personal prices. Price ratio is
    log(item median)-log(user median), or zero if either is missing; the item
    observed flag and user_price_missing distinguish that fallback. Missing
    metadata reduces known affinities rather than becoming a shared category.
    """

    def __init__(
        self, products: pd.DataFrame, purchases: pd.DataFrame, crowd: pd.DataFrame, as_of: date
    ) -> None:
        _require(products, ("product_id", *METADATA), "products")
        if products["product_id"].isna().any() or products["product_id"].duplicated().any():
            raise ValueError("Products need unique nonmissing product_id values")
        self._metadata = {
            row[0]: tuple(
                None if pd.isna(value) or not str(value).strip() else str(value).strip().casefold()
                for value in row[1:]
            )
            for row in products[["product_id", *METADATA]].itertuples(index=False, name=None)
        }
        as_of = pd.Timestamp(as_of).normalize()
        if pd.isna(as_of):
            raise ValueError("as_of must be a valid calendar date")
        history = _past(purchases, "user_id", as_of)
        past = _past(crowd, "shopper_id", as_of)
        items = pd.DataFrame(index=pd.Index(products["product_id"], name="product_id"))
        counts = {}
        for days in (1, 7, 28):
            recent = past[past["purchase_date"] >= as_of - pd.Timedelta(days=days - 1)]
            counts[days] = (
                recent.groupby("product_id", observed=True)
                .size()
                .reindex(items.index, fill_value=0)
            )
            items[f"item_count_{days}d_log"] = np.log1p(counts[days])
        grouped = past.groupby("product_id", observed=True)
        for name, dates in [
            ("last", grouped["purchase_date"].max()),
            ("first", grouped["purchase_date"].min()),
        ]:
            ages = (as_of - dates).dt.days.reindex(items.index).fillna(365).clip(upper=365)
            items[f"item_{name}_sale_days_log"] = np.log1p(ages)
        items["item_trend_7d_28d"] = np.log1p(counts[7]) - np.log1p(counts[28] / 4)
        recent = past[past["purchase_date"] >= as_of - pd.Timedelta(days=27)]
        grouped = recent.groupby("product_id", observed=True)
        items["item_online_share_28d"] = grouped["online"].mean().reindex(items.index).fillna(0)
        median = grouped["price_normalized"].median().reindex(items.index)
        items["item_price_median_28d"] = median.where(np.isfinite(median) & (median > 0), 0)
        items["item_price_observed_28d"] = (items["item_price_median_28d"] > 0).astype(float)
        self._items = items.to_numpy(dtype=float)
        self._position = {pid: i for i, pid in enumerate(items.index)}

        grouped = history.groupby("user_id", observed=True)
        total = grouped.size()
        users = pd.DataFrame(index=total.index)
        users["count"] = np.log1p(total)
        recent = history[history["purchase_date"] >= as_of - pd.Timedelta(days=6)]
        users["recent"] = np.log1p(
            recent.groupby("user_id", observed=True).size().reindex(total.index, fill_value=0)
        )
        users["age"] = np.log1p((as_of - grouped["purchase_date"].max()).dt.days.clip(upper=365))
        users["online"] = grouped["online"].mean().fillna(0)
        median = grouped["price_normalized"].median()
        users["median"] = median.where(np.isfinite(median) & (median > 0), 0)
        self._users = {uid: row.to_numpy(dtype=float) for uid, row in users.iterrows()}
        self._affinities = []
        for i in range(len(METADATA)):
            matches = Counter(
                (uid, values[i])
                for uid, pid in history[["user_id", "product_id"]].itertuples(
                    index=False, name=None
                )
                if (values := self._metadata.get(pid)) is not None and values[i] is not None
            )
            self._affinities.append({key: count / total[key[0]] for key, count in matches.items()})

    def features(self, user_id: int, product_ids: Iterable[int]) -> np.ndarray:
        """Float64 rows in FEATURE_NAMES order, preserving duplicate IDs; unknown IDs fail."""
        ids = list(product_ids)
        if any(pid not in self._position for pid in ids):
            raise ValueError("Candidate product IDs must belong to the supplied catalog")
        item = self._items[[self._position[pid] for pid in ids]]
        user = self._users.get(user_id, np.array([0, 0, np.log1p(365), 0, 0]))
        ratio = np.zeros(len(ids))
        if user[4] > 0:
            known = item[:, 8] > 0
            ratio[known] = np.log(item[known, 7]) - np.log(user[4])
        flags = np.full(len(ids), float(user[4] <= 0))
        affinities = np.array(
            [
                [
                    self._affinities[i].get((user_id, value), 0)
                    for i, value in enumerate(self._metadata[pid])
                ]
                for pid in ids
            ]
        ).reshape(-1, len(METADATA))
        result = np.column_stack((item, ratio, flags, np.tile(user[:4], (len(ids), 1)), affinities))
        if not np.isfinite(result).all():
            raise ValueError("Temporal features must be finite")
        return result
