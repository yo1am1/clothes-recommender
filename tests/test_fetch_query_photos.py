import json
from io import BytesIO

import httpx
import pandas as pd
import pytest
from PIL import Image

from scripts import fetch_photos, fetch_query_photos

ARTICLES = pd.DataFrame(
    {
        "article_id": [108775015, 108775044, 110065001, 111565001, 111586001],
        "prod_name": ["Strap top", "Strap top 2", "OP T-shirt", "Tights", "Socks"],
        "product_type_name": ["Vest top", "Vest top", "T-shirt", "Leggings/Tights", "Socks"],
        "product_group_name": [
            "Garment Upper body",
            "Garment Upper body",
            "Garment Upper body",
            "Garment Lower body",
            "Socks & Tights",  # not a query group
        ],
        "colour_group_name": ["Black", "White", "Blue", "Black", "Grey"],
    }
)


@pytest.fixture
def data_dir(tmp_path):
    pd.DataFrame({"product_id": [108775015]}).to_parquet(tmp_path / "products.parquet")
    ARTICLES.to_csv(tmp_path / "articles.csv", index=False)
    return tmp_path


def _jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 18), "navy").save(output, format="JPEG")
    return output.getvalue()


def test_candidates_exclude_catalog_and_non_garment_groups_and_alternate_groups():
    rows = fetch_query_photos.candidate_articles(ARTICLES, {108775015}, seed=0)
    ids = [row["article_id"] for row in rows]
    assert sorted(ids) == [108775044, 110065001, 111565001]
    # Round-robin: the only lower-body article comes second, before the second upper-body one.
    assert ids[1] == 111565001


def test_fetch_skips_missing_photos_until_count_and_writes_labels(data_dir, monkeypatch):
    missing = fetch_photos.photo_url(111565001)

    def respond(request):
        if str(request.url) == missing:
            return httpx.Response(404)
        return httpx.Response(200, content=_jpeg())

    real_client = httpx.Client
    monkeypatch.setattr(
        fetch_query_photos.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(respond)),
    )
    saved = fetch_query_photos.fetch_query_photos(
        data_dir / "articles.csv", data_dir, count=2, seed=0
    )
    # 111565001 has no photo, so the next candidate takes its place.
    assert {row["photo_id"] for row in saved} == {108775044, 110065001}
    out = data_dir / "query_photos"
    assert json.loads((out / "query_photos.json").read_text()) == saved
    assert {path.name for path in out.glob("*.jpg")} == {f"{r['photo_id']}.jpg" for r in saved}
    assert saved[0].keys() == {"photo_id", "name", "product_type", "colour"}


def test_count_must_be_positive(data_dir):
    with pytest.raises(ValueError, match="count"):
        fetch_query_photos.fetch_query_photos(data_dir / "articles.csv", data_dir, 0, 0)
