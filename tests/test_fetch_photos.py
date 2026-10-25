import hashlib
import json
from io import BytesIO
from pathlib import Path

import httpx
import pandas as pd
import pytest
from PIL import Image

from scripts import fetch_photos


def jpeg() -> bytes:
    output = BytesIO()
    Image.new("RGB", (12, 18), "navy").save(output, format="JPEG")
    return output.getvalue()


def test_url_uses_exact_article_id_and_pinned_source():
    assert fetch_photos.photo_url(130035001) == (
        f"{fetch_photos.SOURCE_REPO}/resolve/{fetch_photos.SOURCE_REVISION}"
        "/data/raw/images/013/0130035001.jpg"
    )


@pytest.mark.parametrize("pid", [0, -1, 10**10, 1.5, "130035001", None])
def test_invalid_ids_are_rejected(pid):
    with pytest.raises(ValueError, match="Invalid H&M article ID"):
        fetch_photos.photo_url(pid)


def test_download_validates_exact_bytes_and_publishes_atomically(tmp_path, monkeypatch):
    content = jpeg()
    target = tmp_path / "130035001.jpg"
    replace = fetch_photos.os.replace
    replacements = []

    def checked_replace(source, destination):
        assert not target.exists()
        assert Path(source).read_bytes() == content
        fetch_photos.validate_jpeg(Path(source))
        replacements.append(destination)
        replace(source, destination)

    def respond(request):
        assert str(request.url) == fetch_photos.photo_url(130035001)
        return httpx.Response(200, content=content)

    monkeypatch.setattr(fetch_photos.os, "replace", checked_replace)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = fetch_photos.download_photo(client, 130035001, tmp_path)
    assert result["status"] == "downloaded"
    assert result["bytes"] == len(content)
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    assert target.read_bytes() == content
    assert replacements == [target]
    assert list(tmp_path.iterdir()) == [target]


def test_existing_valid_jpeg_is_skipped_without_network(tmp_path):
    path = tmp_path / "130035001.jpg"
    path.write_bytes(jpeg())
    before = path.stat().st_mtime_ns

    def no_network(request):
        pytest.fail("Existing valid JPEG must not be downloaded")

    with httpx.Client(transport=httpx.MockTransport(no_network)) as client:
        assert fetch_photos.download_photo(client, 130035001, tmp_path)["status"] == "skipped"
    assert path.stat().st_mtime_ns == before


@pytest.mark.parametrize("content", [b"<html>not a photo</html>", jpeg()[:100]])
def test_invalid_download_does_not_replace_existing_file_or_leave_temporary_files(
    tmp_path, content
):
    path = tmp_path / "130035001.jpg"
    path.write_bytes(b"old invalid file")
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
    ) as client:
        result = fetch_photos.download_photo(client, 130035001, tmp_path)
    assert result["status"] == "failed"
    assert path.read_bytes() == b"old invalid file"
    assert list(tmp_path.iterdir()) == [path]


def test_non_jpeg_is_rejected_even_if_it_decodes(tmp_path):
    content = BytesIO()
    Image.new("RGB", (5, 5)).save(content, format="PNG")
    with httpx.Client(
        transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=content.getvalue())
        )
    ) as client:
        result = fetch_photos.download_photo(client, 1, tmp_path)
    assert result["status"] == "failed"
    assert "JPEG" in result["error"]
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("status, attempts", [(404, 1), (403, 1), (503, 3), (429, 3)])
def test_http_failures_have_bounded_retries_and_no_photo(tmp_path, monkeypatch, status, attempts):
    calls = []
    monkeypatch.setattr(fetch_photos.time, "sleep", lambda delay: None)

    def respond(request):
        calls.append(request)
        return httpx.Response(status)

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = fetch_photos.download_photo(client, 1, tmp_path)
    assert result["status"] == "failed"
    assert len(calls) == attempts
    assert not list(tmp_path.iterdir())


def test_transient_timeout_retries_then_succeeds(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(fetch_photos.time, "sleep", lambda delay: None)

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            raise httpx.ReadTimeout("timed out", request=request)
        return httpx.Response(200, content=jpeg())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        result = fetch_photos.download_photo(client, 1, tmp_path)
    assert result["status"] == "downloaded"
    assert len(calls) == 2


def test_payload_size_is_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(fetch_photos, "MAX_PHOTO_BYTES", 10)
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=jpeg()))
    ) as client:
        result = fetch_photos.download_photo(client, 1, tmp_path)
    assert result["status"] == "failed"
    assert not list(tmp_path.iterdir())


def test_report_records_provenance_failures_and_resumable_skips(tmp_path, monkeypatch):
    pd.DataFrame({"product_id": [3, 1, 2, 1]}).to_parquet(tmp_path / "products.parquet")
    (tmp_path / "images").mkdir()
    (tmp_path / "images" / "1.jpg").write_bytes(jpeg())
    original_client = httpx.Client
    requests = []

    def respond(request):
        requests.append(str(request.url))
        return (
            httpx.Response(404)
            if request.url.path.endswith("3.jpg")
            else httpx.Response(200, content=jpeg())
        )

    monkeypatch.setattr(
        fetch_photos.httpx,
        "Client",
        lambda **kwargs: original_client(transport=httpx.MockTransport(respond)),
    )
    report = fetch_photos.fetch_photos(tmp_path, workers=2)
    assert report["total"] == 3
    assert report["downloaded"] == report["skipped"] == 1
    assert report["failed_ids"] == [3]
    assert len(requests) == 2
    assert "non-commercial" in report["source"]["license_notice"].lower()
    assert report["source"]["revision"] == fetch_photos.SOURCE_REVISION
    assert json.loads((tmp_path / "photo_downloads.json").read_text()) == report
    assert [row["product_id"] for row in report["photos"]] == [1, 2, 3]
    requests.clear()
    resumed = fetch_photos.fetch_photos(tmp_path, workers=1, limit=2)
    assert resumed["skipped"] == 2
    assert requests == []


@pytest.mark.parametrize("options", [{"workers": 0}, {"workers": 9}, {"limit": -1}])
def test_invalid_options_fail_before_reading_or_writing(tmp_path, options):
    with pytest.raises(ValueError):
        fetch_photos.fetch_photos(tmp_path, **options)
    assert not list(tmp_path.iterdir())


def test_atomic_write_cleans_temporary_file_on_failure(tmp_path, monkeypatch):
    def fail_replace(source, target):
        raise OSError("write failed")

    monkeypatch.setattr(fetch_photos.os, "replace", fail_replace)
    with pytest.raises(OSError, match="write failed"):
        fetch_photos.atomic_write(tmp_path / "photo.jpg", jpeg())
    assert not list(tmp_path.iterdir())


def test_large_download_is_saved_downsized(tmp_path):
    large = BytesIO()
    Image.new("RGB", (1166, 1750), "navy").save(large, format="JPEG")
    content = large.getvalue()
    with httpx.Client(
        transport=httpx.MockTransport(lambda request: httpx.Response(200, content=content))
    ) as client:
        result = fetch_photos.download_photo(client, 130035001, tmp_path)
    assert result["sha256"] == hashlib.sha256(content).hexdigest()
    with Image.open(tmp_path / "130035001.jpg") as photo:
        assert photo.format == "JPEG"
        assert photo.width <= fetch_photos.PHOTO_MAX_SIZE[0]
        assert photo.height == fetch_photos.PHOTO_MAX_SIZE[1]
