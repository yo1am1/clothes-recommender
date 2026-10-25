from datetime import date
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from clothes_recommender.catalog import CustomerData, DataFileMissingError, ProductCatalog


def _product_row(product_id: int, product_type: str) -> dict[str, object]:
    return {
        "product_id": product_id,
        "name": f"product {product_id}",
        "product_type": product_type,
        "product_group": "Garment Lower body",
        "section": "Ladieswear",
        "colour": "Black",
        "description": "",
        "price": 19.99,
        "shopify_category": "Apparel & Accessories > Clothing > Pants",
        "shopify_attributes": '{"color": "Black"}',  # stored as JSON in parquet
    }


@pytest.fixture
def catalog(tmp_path: Path) -> ProductCatalog:
    path = tmp_path / "products.parquet"
    rows = [_product_row(3, "Sneakers"), _product_row(1, "Trousers"), _product_row(2, "Sneakers")]
    pd.DataFrame(rows).to_parquet(path)
    return ProductCatalog.from_parquet(path)


@pytest.fixture
def customers(tmp_path: Path) -> CustomerData:
    users_path, purchases_path = tmp_path / "users.parquet", tmp_path / "purchases.parquet"
    pd.DataFrame({"user_id": [2, 1], "age": pd.array([None, 34], dtype="Int64")}).to_parquet(
        users_path
    )
    pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "product_id": [10, 11, 12, 13],
            "purchase_date": pd.to_datetime(
                ["2020-07-10", "2020-09-01", "2020-08-01", "2020-07-05"]
            ).date,
        }
    ).to_parquet(purchases_path)
    return CustomerData.from_parquet(users_path, purchases_path)


def test_catalog_lookup(catalog: ProductCatalog) -> None:
    product = catalog.get(1)

    assert product is not None
    assert (product.product_type, product.price) == ("Trousers", 19.99)
    assert product.shopify_attributes == {"color": "Black"}
    assert catalog.get(404) is None
    assert len(catalog) == 3


def test_get_many_keeps_order_and_skips_unknown_ids(catalog: ProductCatalog) -> None:
    assert [p.product_id for p in catalog.get_many([2, 404, 1])] == [2, 1]


@pytest.mark.parametrize("attributes", ['{"color":', '["Black"]', '{"color": 123}'])
def test_catalog_rejects_invalid_shopify_attributes(attributes: str) -> None:
    row = _product_row(1, "Trousers") | {"shopify_attributes": attributes}
    catalog = ProductCatalog(pd.DataFrame([row]))

    with pytest.raises(ValidationError):
        catalog.get(1)


def test_catalog_before_enrichment_still_loads() -> None:
    row = _product_row(1, "Trousers")
    del row["shopify_category"], row["shopify_attributes"]

    product = ProductCatalog(pd.DataFrame([row])).get(1)

    assert product.shopify_category == "Uncategorized"
    assert product.shopify_attributes == {}


def test_iteration_and_distinct_values(catalog: ProductCatalog) -> None:
    assert sorted(p.product_id for p in catalog) == [1, 2, 3]
    assert catalog.distinct("product_type") == ["Sneakers", "Trousers"]


def test_users_with_unknown_age(customers: CustomerData) -> None:
    assert customers.user_ids() == [1, 2]
    assert customers.get_user(1).age == 34
    assert customers.get_user(2).age is None
    assert customers.get_user(404) is None


def test_purchases_for_returns_most_recent_first(customers: CustomerData) -> None:
    purchases = customers.purchases_for(1)

    assert [p.product_id for p in purchases] == [11, 12, 10]
    assert purchases[0].purchase_date == date(2020, 9, 1)
    assert customers.purchases_for(404) == []


def test_missing_data_file_raises_actionable_error(tmp_path: Path) -> None:
    with pytest.raises(DataFileMissingError, match="prepare_data"):
        ProductCatalog.from_parquet(tmp_path / "missing.parquet")
