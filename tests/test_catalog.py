from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from clothes_recommender.catalog import CustomerData, DataFileMissingError, ProductCatalog


def _product_row(product_id: int, category: str) -> dict[str, object]:
    return {
        "product_id": product_id,
        "name": f"product {product_id}",
        "category": category,
        "brand": "acme",
        "color": "blue",
        "price": 19.99,
        "rating": 4.2,
        "review_count": 12.0,  # the parquet stores counts as floats
        "image_url": f"https://example.com/{product_id}.jpg",
        "product_url": f"https://example.com/{product_id}",
    }


@pytest.fixture
def catalog(tmp_path: Path) -> ProductCatalog:
    path = tmp_path / "products.parquet"
    rows = [_product_row(3, "shoes"), _product_row(1, "jeans"), _product_row(2, "shoes")]
    pd.DataFrame(rows).to_parquet(path)
    return ProductCatalog.from_parquet(path)


@pytest.fixture
def customers(tmp_path: Path) -> CustomerData:
    users_path, purchases_path = tmp_path / "users.csv", tmp_path / "purchases.csv"
    pd.DataFrame(
        {
            "user_id": [2, 1],
            "name": ["Bob", "Ann"],
            "email": ["bob@example.com", "ann@example.com"],
            "city": ["Utrecht", "Amsterdam"],
            "country": ["Netherlands", "Netherlands"],
            "gender": ["M", "F"],
        }
    ).to_csv(users_path, index=False)
    pd.DataFrame(
        {
            "purchase_id": ["P1", "P2", "P3", "P4"],
            "user_id": [1, 1, 1, 2],
            "product_id": [10, 11, 10, 12],
            "purchase_date": ["2025-01-10", "2025-03-01", "2025-02-01", "2025-01-05"],
            "quantity": [1, 2, 1, 1],
        }
    ).to_csv(purchases_path, index=False)
    return CustomerData.from_csv(users_path, purchases_path)


def test_catalog_lookup(catalog: ProductCatalog) -> None:
    product = catalog.get(1)

    assert product is not None
    assert (product.category, product.review_count) == ("jeans", 12)
    assert catalog.get(404) is None
    assert len(catalog) == 3


def test_get_many_keeps_order_and_skips_unknown_ids(catalog: ProductCatalog) -> None:
    assert [p.product_id for p in catalog.get_many([2, 404, 1])] == [2, 1]


def test_iteration_and_categories(catalog: ProductCatalog) -> None:
    assert sorted(p.product_id for p in catalog) == [1, 2, 3]
    assert catalog.categories() == ["jeans", "shoes"]


def test_users(customers: CustomerData) -> None:
    user = customers.get_user(1)

    assert customers.user_ids() == [1, 2]
    assert user is not None
    assert user.name == "Ann"
    assert customers.get_user(404) is None


def test_purchases_for_returns_most_recent_first(customers: CustomerData) -> None:
    purchases = customers.purchases_for(1)

    assert [p.purchase_id for p in purchases] == ["P2", "P3", "P1"]
    assert purchases[0].purchase_date == date(2025, 3, 1)


def test_missing_data_file_raises_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(DataFileMissingError, match="repository root"):
        ProductCatalog.from_parquet(tmp_path / "missing.parquet")
