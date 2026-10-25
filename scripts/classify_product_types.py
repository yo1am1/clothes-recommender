"""Map every H&M product type to a category of Shopify's Standard Product Taxonomy.

Usage: uv run python -m scripts.classify_product_types --articles path/to/articles.csv

There are only ~130 product types, so each gets one reviewable decision instead of
classifying 100k articles one by one:

1. A unique exact name match in a generic apparel/bag branch is accepted as is.
2. Otherwise CLIP text similarity and word overlap shortlist candidate categories,
   and the configured LLM picks one, with a confidence and a reason. Only high-confidence
   decisions are saved; uncertain or invalid choices remain unmapped.

Results go to taxonomy/hm_shopify_categories.json. Entries already in the file are kept,
so hand-reviewed decisions survive re-runs; delete an entry to classify it again.
"""

import argparse
import json
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import httpx
import numpy as np
import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from clothes_recommender.chat.llm import build_chat_model
from clothes_recommender.config import get_settings
from clothes_recommender.embeddings import ClipEncoder
from scripts.shopify_enrichment import CATEGORY_MAPPING_PATH

SHOPIFY_TAXONOMY_VERSION = "2026-05"
TAXONOMY_URL = (
    "https://raw.githubusercontent.com/Shopify/product-taxonomy/"
    f"v{SHOPIFY_TAXONOMY_VERSION}/dist/en/taxonomy.json"
)
TAXONOMY_CACHE = (
    Path(__file__).resolve().parents[1]
    / f"data/raw/shopify/taxonomy-{SHOPIFY_TAXONOMY_VERSION}.json"
)
CANDIDATES_PER_METHOD = 12
EXAMPLES_PER_TYPE = 5

logger = logging.getLogger(__name__)

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You classify an H&M product type into Shopify's Standard Product Taxonomy. "
            "Examples are a small sample, not the entire product type. "
            "Pick a candidate broad enough for the product type and every example article. "
            "Do not infer gender, age, sport, or workwear from a name alone. "
            "Answer 0 if no candidate fits.",
        ),
        (
            "human",
            "Product type: {product_type} (H&M product group: {product_group})\n"
            "Example articles:\n{examples}\n\nCandidates:\n{candidates}",
        ),
    ]
)


@dataclass(frozen=True)
class Category:
    id: str
    full_name: str
    name: str


class CategoryChoice(BaseModel):
    candidate: int = Field(description="Number of the best candidate, or 0 if none fits.")
    confidence: Literal["high", "medium", "low"]
    reason: str = Field(description="One sentence citing the example articles.")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--articles", type=Path, required=True, help="H&M articles table (.csv or .parquet)."
    )
    return parser.parse_args(argv)


def load_taxonomy() -> list[Category]:
    if not TAXONOMY_CACHE.exists():
        TAXONOMY_CACHE.parent.mkdir(parents=True, exist_ok=True)
        response = httpx.get(TAXONOMY_URL, timeout=60, follow_redirects=True)
        response.raise_for_status()
        TAXONOMY_CACHE.write_bytes(response.content)
    taxonomy = json.loads(TAXONOMY_CACHE.read_text())
    return [
        Category(c["id"], c["full_name"], c["name"])
        for vertical in taxonomy["verticals"]
        for c in vertical["categories"]
    ]


def _words(text: str) -> set[str]:
    # Crude singularisation is enough to match "Hoodie" with "Hoodies".
    return {w.removesuffix("s") for w in re.findall(r"[a-z]+", text.lower())}


def exact_match(product_type: str, categories: Sequence[Category]) -> Category | None:
    matches = [c for c in categories if _words(c.name) == _words(product_type)]
    if len(matches) != 1:
        return None
    category = matches[0]
    # A unique name can still mean medical braces or an activewear-only jacket.
    generic_parents = {
        "Apparel & Accessories > Clothing",
        "Apparel & Accessories > Clothing > Clothing Tops",
        "Apparel & Accessories > Clothing > Sleepwear & Loungewear",
        "Apparel & Accessories > Clothing Accessories",
        "Apparel & Accessories > Clothing Accessories > Hats",
        "Apparel & Accessories > Jewelry",
        "Apparel & Accessories > Shoes",
        "Luggage & Bags",
    }
    return category if category.full_name.rpartition(" > ")[0] in generic_parents else None


def shortlist(
    query: str,
    query_vector: np.ndarray,
    categories: Sequence[Category],
    category_vectors: np.ndarray,
) -> list[Category]:
    """Union of the best categories by CLIP text similarity and by word overlap."""
    dense = np.argsort(-(category_vectors @ query_vector))[:CANDIDATES_PER_METHOD]
    query_words = _words(query)
    overlap = [len(query_words & _words(c.full_name)) for c in categories]
    lexical = np.argsort(-np.array(overlap), kind="stable")[:CANDIDATES_PER_METHOD]
    ranked = dict.fromkeys([*dense.tolist(), *lexical.tolist()])
    return [categories[i] for i in ranked]


def classify_with_llm(
    chat_model: BaseChatModel,
    product_type: str,
    product_group: str,
    examples: Sequence[str],
    candidates: Sequence[Category],
) -> dict[str, str] | None:
    chain = PROMPT | chat_model.with_structured_output(CategoryChoice, method="function_calling")
    choice = chain.invoke(
        {
            "product_type": product_type,
            "product_group": product_group,
            "examples": "\n".join(f"- {example}" for example in examples),
            "candidates": "\n".join(f"{i}. {c.full_name}" for i, c in enumerate(candidates, 1)),
        }
    )
    if choice.confidence != "high" or not 1 <= choice.candidate <= len(candidates):
        return None
    category = candidates[choice.candidate - 1]
    return {
        "category_id": category.id,
        "category": category.full_name,
        "method": "llm",
        "confidence": choice.confidence,
        "reason": choice.reason,
    }


def main(argv: Sequence[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    read = pd.read_parquet if args.articles.suffix == ".parquet" else pd.read_csv
    articles = read(args.articles)
    categories = load_taxonomy()
    mapping: dict[str, dict[str, str]] = (
        json.loads(CATEGORY_MAPPING_PATH.read_text()) if CATEGORY_MAPPING_PATH.exists() else {}
    )

    pending = sorted(set(articles["product_type_name"]) - set(mapping))
    logger.info("%d product types to classify, %d already mapped", len(pending), len(mapping))
    encoder = None
    category_vectors = None
    chat_model = None

    for product_type in pending:
        rows = articles[articles["product_type_name"] == product_type]
        product_group = rows["product_group_name"].mode().iloc[0]
        if category := exact_match(product_type, categories):
            mapping[product_type] = {
                "category_id": category.id,
                "category": category.full_name,
                "method": "exact",
                "confidence": "high",
                "reason": "Unique category with the same name.",
            }
        else:
            if encoder is None:
                encoder = ClipEncoder(get_settings().clip_model_name)
                category_vectors = encoder.encode_texts([c.full_name for c in categories])
                chat_model = build_chat_model(get_settings())
            descriptions = (
                rows["prod_name"].fillna("") + ": " + rows["detail_desc"].fillna("")
            ).drop_duplicates()
            # Spread the small sample across the catalog instead of its first variants.
            examples = descriptions.iloc[
                np.linspace(
                    0, len(descriptions) - 1, min(EXAMPLES_PER_TYPE, len(descriptions)), dtype=int
                )
            ]
            query = f"{product_type} {product_group}"
            candidates = shortlist(
                query, encoder.encode_texts([query])[0], categories, category_vectors
            )
            decision = classify_with_llm(
                chat_model, product_type, product_group, examples.tolist(), candidates
            )
            if decision is None:
                logger.warning("No fitting category for %r; left unmapped.", product_type)
                continue
            mapping[product_type] = decision
        logger.info("%s -> %s", product_type, mapping[product_type]["category"])
        # Written after every decision so an interrupted run loses nothing.
        CATEGORY_MAPPING_PATH.parent.mkdir(exist_ok=True)
        CATEGORY_MAPPING_PATH.write_text(json.dumps(dict(sorted(mapping.items())), indent=2) + "\n")


if __name__ == "__main__":
    main()
