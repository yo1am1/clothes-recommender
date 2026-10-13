"""Builders for tiny, hand-made catalogs and purchase histories."""

from collections.abc import Callable, Sequence
from datetime import date

import pandas as pd
import pytest

from clothes_recommender.catalog import CustomerData, ProductCatalog

# (product_id, category, rating, review_count)
ProductSpec = tuple[int, str, float, int]
# (user_id, product_id, purchase_date)
PurchaseSpec = tuple[int, int, date]


@pytest.fixture
def make_catalog() -> Callable[[Sequence[ProductSpec]], ProductCatalog]:
    def build(specs: Sequence[ProductSpec]) -> ProductCatalog:
        rows = [
            {
                "product_id": product_id,
                "name": f"product {product_id}",
                "category": category,
                "brand": "acme",
                "color": "black",
                "price": 10.0,
                "rating": rating,
                "review_count": review_count,
                "image_url": f"https://example.com/{product_id}.jpg",
                "product_url": f"https://example.com/{product_id}",
            }
            for product_id, category, rating, review_count in specs
        ]
        return ProductCatalog(pd.DataFrame(rows))

    return build


@pytest.fixture
def make_customers() -> Callable[[Sequence[int], Sequence[PurchaseSpec]], CustomerData]:
    def build(user_ids: Sequence[int], purchases: Sequence[PurchaseSpec]) -> CustomerData:
        users = pd.DataFrame(
            {
                "user_id": list(user_ids),
                "name": [f"user {u}" for u in user_ids],
                "email": [f"user{u}@example.com" for u in user_ids],
                "city": "Amsterdam",
                "country": "Netherlands",
                "gender": "F",
            }
        )
        purchase_rows = pd.DataFrame(
            [
                {
                    "purchase_id": f"P{i}",
                    "user_id": user_id,
                    "product_id": product_id,
                    "purchase_date": purchase_date.isoformat(),
                    "quantity": 1,
                }
                for i, (user_id, product_id, purchase_date) in enumerate(purchases)
            ],
            columns=["purchase_id", "user_id", "product_id", "purchase_date", "quantity"],
        )
        return CustomerData(users, purchase_rows)

    return build
