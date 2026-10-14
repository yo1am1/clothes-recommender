"""Builders for tiny, hand-made catalogs and purchase histories."""

from collections.abc import Callable, Sequence
from datetime import date

import pandas as pd
import pytest

from clothes_recommender.catalog import CustomerData, ProductCatalog

# (product_id, product_type)
ProductSpec = tuple[int, str]
# (user_id, product_id, purchase_date)
PurchaseSpec = tuple[int, int, date]


@pytest.fixture
def make_catalog() -> Callable[[Sequence[ProductSpec]], ProductCatalog]:
    def build(specs: Sequence[ProductSpec]) -> ProductCatalog:
        rows = [
            {
                "product_id": product_id,
                "name": f"product {product_id}",
                "product_type": product_type,
                "section": "Ladieswear",
                "colour": "Black",
                "description": "",
                "price": 10.0,
            }
            for product_id, product_type in specs
        ]
        return ProductCatalog(pd.DataFrame(rows))

    return build


@pytest.fixture
def make_customers() -> Callable[[Sequence[int], Sequence[PurchaseSpec]], CustomerData]:
    def build(user_ids: Sequence[int], purchases: Sequence[PurchaseSpec]) -> CustomerData:
        users = pd.DataFrame({"user_id": list(user_ids), "age": 30})
        purchase_rows = pd.DataFrame(purchases, columns=["user_id", "product_id", "purchase_date"])
        return CustomerData(users, purchase_rows)

    return build
