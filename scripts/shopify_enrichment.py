"""Shopify Standard Product Taxonomy enrichment for H&M articles.

Categories come from the persisted product-type mapping that
scripts/classify_product_types.py writes. Attributes are mapped deterministically, and
only where unambiguous: a value missing below (e.g. H&M's "Metal" colour, or the
mixed-gender Divided and Sport sections) stays unmapped rather than guessed.
"""

import json
from pathlib import Path

import pandas as pd

CATEGORY_MAPPING_PATH = Path(__file__).resolve().parents[1] / "taxonomy/hm_shopify_categories.json"
# Shopify's own category for products that fit nowhere else.
UNCATEGORIZED = "Uncategorized"

# H&M perceived_colour_master_name -> Shopify "color" value
COLOR = {
    "Beige": "Beige",
    "Black": "Black",
    "Blue": "Blue",
    "Bluish Green": "Green",
    "Brown": "Brown",
    "Green": "Green",
    "Grey": "Gray",
    "Khaki green": "Green",
    "Lilac Purple": "Purple",
    "Orange": "Orange",
    "Pink": "Pink",
    "Red": "Red",
    "White": "White",
    "Yellow": "Yellow",
    "Yellowish Green": "Green",
}

# H&M graphical_appearance_name -> Shopify "pattern" value
PATTERN = {
    "Argyle": "Diamond",
    "Check": "Checkered",
    "Dot": "Dots",
    "Lace": "Lace",
    "Solid": "Solid",
    "Stripe": "Striped",
}

# H&M index_group_name -> Shopify "target-gender" value
TARGET_GENDER = {"Ladieswear": "Female", "Menswear": "Male"}

# H&M index_name -> Shopify "age-group" value
AGE_GROUP = {
    "Baby Sizes 50-98": "Babies",
    "Children Sizes 92-140": "Kids",
    "Children Sizes 134-170": "Kids",
    "Children Accessories, Swimwear": "Kids",
    "Ladieswear": "Adults",
    "Ladies Accessories": "Adults",
    "Lingeries/Tights": "Adults",
    "Menswear": "Adults",
}

_SOURCES = {
    "color": ("perceived_colour_master_name", COLOR),
    "pattern": ("graphical_appearance_name", PATTERN),
    "target-gender": ("index_group_name", TARGET_GENDER),
    "age-group": ("index_name", AGE_GROUP),
}


def shopify_attributes(article: pd.Series) -> dict[str, str]:
    """Shopify attribute handle -> value for one H&M article row, mapped values only."""
    attributes = {}
    for handle, (column, values) in _SOURCES.items():
        if (value := values.get(article[column])) is not None:
            attributes[handle] = value
    return attributes


def shopify_categories(product_types: pd.Series) -> pd.Series:
    """Full Shopify category path for each H&M product type."""
    mapping = json.loads(CATEGORY_MAPPING_PATH.read_text())
    return product_types.map(lambda t: mapping[t]["category"] if t in mapping else UNCATEGORIZED)
