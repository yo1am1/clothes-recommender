"""Offline checks for deterministic Shopify enrichment and the persisted mapping."""

import json
from pathlib import Path

import pandas as pd
import pytest

from scripts import shopify_enrichment as enrichment

ROOT = Path(__file__).resolve().parents[1]


def test_categories_preserve_index_and_fall_back_from_another_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    product_types = pd.Series(["Trousers", "Unknown", None, "not a type"], index=[4, 8, 9, 12])
    result = enrichment.shopify_categories(product_types)
    assert result.index.equals(product_types.index)
    assert result.tolist() == [
        "Apparel & Accessories > Clothing > Pants",
        *[enrichment.UNCATEGORIZED] * 3,
    ]


def test_attributes_map_only_unambiguous_source_values():
    article = pd.Series(
        {
            "perceived_colour_master_name": "Grey",
            "graphical_appearance_name": "Stripe",
            "index_group_name": "Ladieswear",
            "index_name": "Ladieswear",
        }
    )
    assert enrichment.shopify_attributes(article) == {
        "color": "Gray",
        "pattern": "Striped",
        "target-gender": "Female",
        "age-group": "Adults",
    }
    article[:] = ["Metal", "All over pattern", "Divided", "Divided"]
    assert enrichment.shopify_attributes(article) == {}
    article[:] = [pd.NA, None, "Sport", "Sport"]
    assert enrichment.shopify_attributes(article) == {}


def cached_taxonomy():
    path = ROOT / "data/raw/shopify/taxonomy-2026-05.json"
    if not path.exists():
        pytest.skip("Local taxonomy cache is not available; no download in offline tests.")
    return json.loads(path.read_text())


def test_all_deterministic_attribute_values_are_allowed_by_cached_taxonomy():
    allowed = {
        attribute["handle"]: {value["name"] for value in attribute["values"]}
        for attribute in cached_taxonomy()["attributes"]
    }
    for handle, (_, mapping) in enrichment._SOURCES.items():
        assert set(mapping.values()) <= allowed[handle]


def test_persisted_category_ids_and_paths_match_cached_taxonomy():
    categories = {
        category["id"]: category["full_name"]
        for vertical in cached_taxonomy()["verticals"]
        for category in vertical["categories"]
    }
    mapping = json.loads(enrichment.CATEGORY_MAPPING_PATH.read_text())
    for entry in mapping.values():
        assert categories[entry["category_id"]] == entry["category"]


def test_actual_hm_type_coverage_and_ambiguous_fallback():
    path = ROOT / "data/raw/hf/articles/train-00000-of-00001.parquet"
    if not path.exists():
        pytest.skip("Local H&M articles are not available.")
    types = set(pd.read_parquet(path, columns=["product_type_name"])["product_type_name"])
    mapping = json.loads(enrichment.CATEGORY_MAPPING_PATH.read_text())
    assert set(mapping) <= types
    unmapped = {"Unknown", "Accessories set", "Other accessories", "Soft Toys"}
    assert types - set(mapping) == unmapped
    assert set(enrichment.shopify_categories(pd.Series(sorted(unmapped)))) == {
        enrichment.UNCATEGORIZED
    }
