import io
import json
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts import shopify_enrichment
from scripts.prepare_data import HM_PRICE_SCALE, PHOTO_MAX_SIZE, co_purchases, photo_member, prepare

ARTICLES = pd.DataFrame(
    {
        "article_id": [108775015, 108775044, 110065001],
        "product_code": [108775, 108775, 110065],
        "department_name": ["Jersey Basic", "Jersey Basic", "Jersey"],
        "prod_name": ["Strap top", "Strap top", "OP T-shirt"],
        "product_type_name": ["Vest top", "Vest top", "T-shirt"],
        "product_group_name": ["Garment Upper body"] * 3,
        "index_group_name": ["Ladieswear", "Ladieswear", "Menswear"],
        "colour_group_name": ["Black", "White", "Dark Blue"],
        "perceived_colour_master_name": ["Black", "White", "Blue"],
        "graphical_appearance_name": ["Solid"] * 3,
        "index_name": ["Ladieswear", "Ladieswear", "Menswear"],
        "detail_desc": ["Jersey top.", None, "Cotton T-shirt."],
    }
)
CUSTOMERS = pd.DataFrame({"customer_id": ["a", "b", "c", "d"], "age": [24, None, 51, 33]})
# Article 108775044 has no photo in the archive, so it must be dropped with its purchases.
PHOTOS = [108775015, 110065001]


def _transactions() -> pd.DataFrame:
    rows = [
        *[("2020-09-20", "a", 108775015, 0.0508)] * 4,
        ("2020-09-21", "a", 108775044, 0.0305),
        ("2020-09-22", "a", 110065001, 0.0152),
        *[("2020-09-17", "b", 110065001, 0.0169)] * 5,
        # Two shared baskets, but only four purchases: c is not sampled.
        ("2020-09-18", "c", 108775015, 0.0508),
        ("2020-09-18", "c", 110065001, 0.0152),
        ("2020-09-19", "c", 108775015, 0.0508),
        ("2020-09-19", "c", 110065001, 0.0152),
        *[("2020-08-01", "d", 110065001, 0.0152)] * 9,  # active, but before the 1-week window
    ]
    return pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id", "price"]).assign(
        sales_channel_id=2
    )


@pytest.fixture(autouse=True)
def category_mapping(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mapping = tmp_path / "categories.json"
    mapping.write_text(
        json.dumps(
            {
                product_type: {"category": "Apparel & Accessories > Clothing > Shirts & Tops"}
                for product_type in ARTICLES["product_type_name"].unique()
            }
        )
    )
    monkeypatch.setattr(shopify_enrichment, "CATEGORY_MAPPING_PATH", mapping)


@pytest.fixture
def archive_factory(tmp_path: Path):
    def build(*, articles=ARTICLES, customers=CUSTOMERS, transactions=None, photos=PHOTOS):
        path = tmp_path / "hm.zip"
        if transactions is None:
            transactions = _transactions()
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("articles.csv", articles.to_csv(index=False))
            archive.writestr("customers.csv", customers.to_csv(index=False))
            archive.writestr("transactions_train.csv", transactions.to_csv(index=False))
            for article_id in photos:
                photo = io.BytesIO()
                Image.new("RGB", (1166, 1750), "navy").save(photo, format="JPEG")
                archive.writestr(photo_member(article_id), photo.getvalue())
        return path

    return build


@pytest.fixture
def archive(archive_factory) -> Path:
    return archive_factory()


@pytest.fixture
def data_dir(archive: Path, tmp_path: Path) -> Path:
    out = tmp_path / "data"
    prepare(archive, out, weeks=1, customer_count=10, seed=0)
    return out


def test_samples_only_customers_active_in_the_window(data_dir: Path) -> None:
    users = pd.read_parquet(data_dir / "users.parquet")

    assert users["user_id"].tolist() == [1, 2]  # customers a and b, in id order
    assert users["age"].tolist() == [24, pd.NA]


def test_drops_articles_without_photos_and_their_purchases(data_dir: Path) -> None:
    products = pd.read_parquet(data_dir / "products.parquet").set_index("product_id")
    purchases = pd.read_parquet(data_dir / "purchases.parquet")

    assert sorted(products.index) == PHOTOS
    assert set(purchases["product_id"]) == set(PHOTOS)
    assert len(purchases) == 4 + 1 + 5


def test_maps_hm_columns_and_scales_median_price(data_dir: Path) -> None:
    top = pd.read_parquet(data_dir / "products.parquet").set_index("product_id").loc[108775015]

    assert (top["name"], top["product_type"], top["section"], top["colour"]) == (
        "Strap top",
        "Vest top",
        "Ladieswear",
        "Black",
    )
    assert top["price"] == round(0.0508 * HM_PRICE_SCALE, 2)


def test_extracts_downsized_photos(data_dir: Path) -> None:
    with Image.open(data_dir / "images" / "110065001.jpg") as photo:
        assert photo.width <= PHOTO_MAX_SIZE[0] and photo.height <= PHOTO_MAX_SIZE[1]
    assert not (data_dir / "images" / "108775044.jpg").exists()


def test_enriches_products_using_isolated_mapping(data_dir: Path) -> None:
    top = pd.read_parquet(data_dir / "products.parquet").iloc[0]

    assert top["shopify_category"] == "Apparel & Accessories > Clothing > Shirts & Tops"
    assert json.loads(top["shopify_attributes"]) == {
        "color": "Black",
        "pattern": "Solid",
        "target-gender": "Female",
        "age-group": "Adults",
    }


def test_counts_co_purchases_from_unsampled_shoppers(data_dir: Path) -> None:
    links = pd.read_parquet(data_dir / "co_purchases.parquet")

    assert links.to_dict("records") == [
        {"product_id": 108775015, "other_id": 110065001, "count": 2, "lift": 1.0}
    ]


def test_keeps_every_shoppers_catalog_purchases_anonymously(data_dir: Path) -> None:
    shoppers = pd.read_parquet(data_dir / "shopper_purchases.parquet")

    assert list(shoppers.columns) == [
        "shopper_id",
        "product_id",
        "purchase_date",
        "price_normalized",
        "sales_channel_id",
    ]
    assert shoppers["shopper_id"].dtype == "int32"
    # Sampled a and b plus unsampled c, inside the window and the photographed catalog only.
    assert shoppers["shopper_id"].nunique() == 3
    assert set(shoppers["product_id"]) == set(PHOTOS)
    assert len(shoppers) == 4 + 1 + 5 + 4
    assert pd.to_datetime(shoppers["purchase_date"]).min() >= pd.Timestamp("2020-09-16")


@pytest.mark.parametrize(
    "article_ids, rows",
    [
        ([], []),
        ([1, 2], []),
        ([], [("a", "2020-09-20", 3)]),
        ([1, 2], [("a", "2020-09-20", 3)]),
        ([1], [("a", "2020-09-20", 1)]),
    ],
)
def test_co_purchases_without_pairs(article_ids, rows) -> None:
    transactions = pd.DataFrame(rows, columns=["customer_id", "t_dat", "article_id"])

    links = co_purchases(transactions, article_ids)

    assert links.empty
    assert links.columns.tolist() == ["product_id", "other_id", "count", "lift"]
    assert links.dtypes.astype(str).tolist() == ["int64", "int64", "int64", "float64"]


def test_co_purchases_counts_distinct_baskets_and_uses_all_baskets_for_lift() -> None:
    transactions = pd.DataFrame(
        [
            ("a", "2020-09-20", 1),
            ("a", "2020-09-20", 1),
            ("a", "2020-09-20", 2),
            ("a", "2020-09-20", 3),  # A pair seen only once must be excluded.
            ("a", "2020-09-21", 1),
            ("a", "2020-09-21", 2),
            ("b", "2020-09-20", 1),
            ("c", "2020-09-20", 2),
            ("d", "2020-09-20", 4),  # A basket outside the catalog still counts for lift.
        ],
        columns=["customer_id", "t_dat", "article_id"],
    )

    assert co_purchases(transactions, [1, 2, 3]).to_dict("records") == [
        {"product_id": 1, "other_id": 2, "count": 2, "lift": 1.111}
    ]


@pytest.mark.parametrize(
    "options",
    [
        {"weeks": 0},
        {"weeks": -1},
        {"customer_count": 0},
        {"customer_count": -1},
        {"seed": -1},
    ],
)
def test_rejects_invalid_preparation_options(tmp_path: Path, options) -> None:
    arguments = {"weeks": 1, "customer_count": 10, "seed": 0} | options
    out = tmp_path / "data"

    with pytest.raises(ValueError, match=r"positive|nonnegative"):
        prepare(tmp_path / "missing.zip", out, **arguments)
    assert not out.exists()


@pytest.mark.parametrize(
    "transactions",
    [
        _transactions().iloc[:0],
        _transactions().assign(t_dat="2020-01-01"),
        _transactions().iloc[:1],
    ],
)
def test_rejects_window_without_eligible_customers(archive_factory, tmp_path, transactions) -> None:
    archive = archive_factory(transactions=transactions)
    out = tmp_path / "data"

    with pytest.raises(ValueError, match="No eligible customers"):
        prepare(archive, out, weeks=1, customer_count=10, seed=0)
    assert not out.exists()


@pytest.mark.parametrize("price", [None, float("inf"), -0.1, 0])
def test_rejects_invalid_transaction_prices(archive_factory, tmp_path, price) -> None:
    transactions = _transactions().assign(price=price)
    archive = archive_factory(transactions=transactions)

    with pytest.raises(ValueError, match="finite positive prices"):
        prepare(archive, tmp_path / "data", weeks=1, customer_count=10, seed=0)
    assert not (tmp_path / "data").exists()


def test_rejects_catalog_without_photos(archive_factory, tmp_path) -> None:
    archive = archive_factory(photos=[999999001])  # the archive has photos, none of these

    with pytest.raises(ValueError, match="No photos"):
        prepare(archive, tmp_path / "data", weeks=1, customer_count=10, seed=0)
    assert not (tmp_path / "data").exists()


@pytest.mark.parametrize("table", ["articles", "customers"])
@pytest.mark.parametrize("invalid", ["missing", "duplicate"])
def test_rejects_inconsistent_metadata(archive_factory, tmp_path, table, invalid) -> None:
    frame = ARTICLES if table == "articles" else CUSTOMERS
    frame = frame.iloc[1:] if invalid == "missing" else pd.concat([frame, frame.iloc[:1]])
    archive = archive_factory(**{table: frame})

    with pytest.raises(ValueError, match="exactly one row"):
        prepare(archive, tmp_path / "data", weeks=1, customer_count=10, seed=0)
    assert not (tmp_path / "data").exists()


def test_drops_customers_left_without_photographed_purchases(archive_factory, tmp_path) -> None:
    archive = archive_factory(photos=[108775015])
    out = tmp_path / "data"
    prepare(archive, out, weeks=1, customer_count=10, seed=0)

    users = pd.read_parquet(out / "users.parquet")
    purchases = pd.read_parquet(out / "purchases.parquet")
    assert users["user_id"].tolist() == [1]
    assert set(users["user_id"]) == set(purchases["user_id"])


@pytest.mark.parametrize("column", ["customer_id", "article_id"])
def test_rejects_missing_transaction_ids(archive_factory, tmp_path, column) -> None:
    transactions = _transactions().astype({"article_id": "Int64"})
    transactions.loc[0, column] = None
    archive = archive_factory(transactions=transactions)

    with pytest.raises(ValueError, match="customer/article IDs"):
        prepare(archive, tmp_path / "data", weeks=1, customer_count=10, seed=0)
    assert not (tmp_path / "data").exists()


def test_reads_unzipped_directory_without_photos(archive_factory, tmp_path) -> None:
    raw = tmp_path / "raw"
    with zipfile.ZipFile(archive_factory(photos=[])) as archive:
        archive.extractall(raw)
    out = tmp_path / "data"
    prepare(raw, out, weeks=1, customer_count=10, seed=0)

    products = pd.read_parquet(out / "products.parquet")
    # Photos are fetched later, so no article is dropped for lacking one.
    assert sorted(products["product_id"]) == sorted(ARTICLES["article_id"])
    assert not (out / "images").exists()


def test_reads_unzipped_directory_with_photos(archive_factory, tmp_path) -> None:
    raw = tmp_path / "raw"
    with zipfile.ZipFile(archive_factory()) as archive:
        archive.extractall(raw)
    out = tmp_path / "data"
    prepare(raw, out, weeks=1, customer_count=10, seed=0)

    assert sorted(pd.read_parquet(out / "products.parquet")["product_id"]) == PHOTOS
    assert sorted(path.stem for path in (out / "images").iterdir()) == [str(p) for p in PHOTOS]


def test_preserves_family_metadata_and_event_observations(data_dir: Path) -> None:
    products = pd.read_parquet(data_dir / "products.parquet").set_index("product_id")
    assert products.loc[108775015, "product_code"] == 108775
    assert products.loc[108775015, "department_name"] == "Jersey Basic"
    assert products.loc[110065001, "index_name"] == "Menswear"
    for filename in ["purchases.parquet", "shopper_purchases.parquet"]:
        events = pd.read_parquet(data_dir / filename)
        assert set(events["sales_channel_id"]) == {2}
        assert events["sales_channel_id"].dtype == "int8"
        assert set(events.loc[events["product_id"] == 108775015, "price_normalized"]) == {0.0508}
        assert events.loc[events["product_id"] == 110065001, "price_normalized"].nunique() == 2
        assert "customer_id" not in events.columns


def test_crosswalk_matches_crowd_histories_after_sampling_and_photo_filtering(
    archive_factory, tmp_path
) -> None:
    # A single sampled customer has user_id=1, but never crowd shopper_id=1 here.
    transactions = pd.concat(
        [
            _transactions(),
            pd.DataFrame(
                [("2020-09-20", "0", 108775015, 0.03, 1)],
                columns=["t_dat", "customer_id", "article_id", "price", "sales_channel_id"],
            ),
        ]
    )
    archive = archive_factory(transactions=transactions)
    out = tmp_path / "data"
    prepare(archive, out, weeks=1, customer_count=1, seed=0)
    users = pd.read_parquet(out / "users.parquet")
    sampled = pd.read_parquet(out / "purchases.parquet")
    crowd = pd.read_parquet(out / "shopper_purchases.parquet")
    assert users["user_id"].tolist() == [1]
    assert users["shopper_id"].tolist() == [2]  # customer b, after 0 and a
    assert users["shopper_id"].dtype == "int32"
    assert "customer_id" not in users.columns
    columns = ["product_id", "purchase_date", "price_normalized", "sales_channel_id"]
    pd.testing.assert_frame_equal(
        sampled[columns].reset_index(drop=True),
        crowd.loc[crowd["shopper_id"] == users.iloc[0]["shopper_id"], columns].reset_index(
            drop=True
        ),
    )


@pytest.mark.parametrize("channel", [None, 0, 3])
def test_rejects_invalid_sales_channels(archive_factory, tmp_path, channel) -> None:
    archive = archive_factory(transactions=_transactions().assign(sales_channel_id=channel))
    with pytest.raises(ValueError, match="Sales channel"):
        prepare(archive, tmp_path / "data", weeks=1, customer_count=10, seed=0)
    assert not (tmp_path / "data").exists()


def test_cli_prepares_to_explicit_experiment_directory(archive_factory, tmp_path) -> None:
    from scripts.prepare_data import main

    out = tmp_path / "experiment"
    main(
        [
            "--archive",
            str(archive_factory()),
            "--data-dir",
            str(out),
            "--weeks",
            "1",
            "--customers",
            "1",
        ]
    )
    assert (out / "users.parquet").exists()


def test_enriched_preparation_remains_compatible_with_serving(data_dir: Path) -> None:
    from clothes_recommender.catalog import CustomerData, ProductCatalog, load_shopper_purchases
    from clothes_recommender.recommenders.collaborative import CollaborativeRecommender
    from clothes_recommender.recommenders.popularity import PopularityRecommender

    catalog = ProductCatalog.from_parquet(data_dir / "products.parquet")
    customers = CustomerData.from_parquet(
        data_dir / "users.parquet", data_dir / "purchases.parquet"
    )
    shoppers = load_shopper_purchases(data_dir / "shopper_purchases.parquet", customers)
    popular = PopularityRecommender(shoppers)
    collaborative = CollaborativeRecommender(customers, catalog, shoppers, popular)
    assert set(customers.get_user(1).model_dump()) == {"user_id", "age"}
    assert "shopper_id" not in customers.purchases_for(1)[0].model_dump()
    assert "price_normalized" not in customers.purchases_for(1)[0].model_dump()
    assert "product_code" not in catalog.get(108775015).model_dump()
    assert collaborative.recommend(2, 2)
