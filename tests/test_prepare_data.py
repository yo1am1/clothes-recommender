import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image

from scripts.prepare_data import HM_PRICE_SCALE, PHOTO_MAX_SIZE, photo_member, prepare

ARTICLES = pd.DataFrame(
    {
        "article_id": [108775015, 108775044, 110065001],
        "prod_name": ["Strap top", "Strap top", "OP T-shirt"],
        "product_type_name": ["Vest top", "Vest top", "T-shirt"],
        "index_group_name": ["Ladieswear", "Ladieswear", "Menswear"],
        "colour_group_name": ["Black", "White", "Dark Blue"],
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
        *[("2020-09-18", "c", 108775015, 0.0508)] * 4,  # only 4 purchases: not sampled
        *[("2020-08-01", "d", 110065001, 0.0152)] * 9,  # active, but before the 1-week window
    ]
    return pd.DataFrame(rows, columns=["t_dat", "customer_id", "article_id", "price"]).assign(
        sales_channel_id=2
    )


@pytest.fixture
def archive(tmp_path: Path) -> Path:
    path = tmp_path / "hm.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("articles.csv", ARTICLES.to_csv(index=False))
        archive.writestr("customers.csv", CUSTOMERS.to_csv(index=False))
        archive.writestr("transactions_train.csv", _transactions().to_csv(index=False))
        for article_id in PHOTOS:
            photo = io.BytesIO()
            Image.new("RGB", (1166, 1750), "navy").save(photo, format="JPEG")
            archive.writestr(photo_member(article_id), photo.getvalue())
    return path


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
