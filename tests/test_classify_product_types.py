"""Classification boundary tests: fake models only, with no downloads or API calls."""

import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from langchain_core.runnables import RunnableLambda

from scripts import classify_product_types as classifier

SHIRTS = classifier.Category(
    "shirts", "Apparel & Accessories > Clothing > Clothing Tops > Shirts", "Shirts"
)


@pytest.mark.parametrize("name", ["Shirt", "SHIRTS"])
def test_exact_match_accepts_unique_generic_apparel(name):
    assert classifier.exact_match(name, [SHIRTS]) == SHIRTS


def test_exact_match_rejects_duplicate_names_and_context_specific_names():
    duplicate = classifier.Category("child-shirts", "Children > Shirts", "Shirts")
    assert classifier.exact_match("Shirt", [SHIRTS, duplicate]) is None
    for name, path in [
        ("Braces", "Health & Beauty > Health Care > Supports & Braces"),
        ("Jackets", "Apparel & Accessories > Clothing > Activewear > Jackets"),
        ("Side Tables", "Furniture > Outdoor Furniture > Outdoor Tables > Side Tables"),
    ]:
        category = classifier.Category("specialized", path, name)
        assert classifier.exact_match(name, [category]) is None
    assert classifier.exact_match("not a category", [SHIRTS]) is None


class FakeChatModel:
    def __init__(self, candidate, confidence):
        self.choice = classifier.CategoryChoice(
            candidate=candidate, confidence=confidence, reason="Offline test choice."
        )

    def with_structured_output(self, schema, *, method):
        assert schema is classifier.CategoryChoice
        assert method == "function_calling"
        return RunnableLambda(lambda _: self.choice)


@pytest.mark.parametrize(
    ("candidate", "confidence", "accepted"),
    [
        (1, "high", True),
        (0, "high", False),
        (-1, "high", False),
        (2, "high", False),
        (1, "low", False),
        (1, "medium", False),
    ],
)
def test_llm_choice_must_be_in_shortlist_and_confident(candidate, confidence, accepted):
    decision = classifier.classify_with_llm(
        FakeChatModel(candidate, confidence), "Shirt", "Upper body", ["Cotton shirt"], [SHIRTS]
    )
    if accepted:
        assert decision == {
            "category_id": SHIRTS.id,
            "category": SHIRTS.full_name,
            "method": "llm",
            "confidence": "high",
            "reason": "Offline test choice.",
        }
    else:
        assert decision is None


def test_shortlist_merges_dense_and_lexical_candidates_without_duplicates(monkeypatch):
    monkeypatch.setattr(classifier, "CANDIDATES_PER_METHOD", 1)
    unrelated = classifier.Category("other", "Other > Unrelated", "Unrelated")
    assert classifier.shortlist(
        "Shirt", np.array([1.0, 0.0]), [SHIRTS, unrelated], np.array([[0.0, 1.0], [1.0, 0.0]])
    ) == [unrelated, SHIRTS]


def test_main_preserves_reviewed_decisions_and_avoids_models_for_exact_matches(
    tmp_path, monkeypatch
):
    mapping_path = tmp_path / "mapping.json"
    reviewed = {
        "category_id": "reviewed-id",
        "category": "Reviewed path",
        "method": "human",
        "confidence": "high",
        "reason": "Existing review.",
        "reviewed": True,
    }
    mapping_path.write_text(json.dumps({"Reviewed type": reviewed}))
    articles = tmp_path / "articles.csv"
    pd.DataFrame(
        {"product_type_name": ["Reviewed type", "Shirt"], "product_group_name": ["Upper"] * 2}
    ).to_csv(articles, index=False)
    monkeypatch.setattr(classifier, "CATEGORY_MAPPING_PATH", mapping_path)
    monkeypatch.setattr(classifier, "load_taxonomy", lambda: [SHIRTS])

    def no_models(*args, **kwargs):
        pytest.fail("Exact-only and already-mapped reruns must not initialize any model.")

    monkeypatch.setattr(classifier, "ClipEncoder", no_models)
    monkeypatch.setattr(classifier, "build_chat_model", no_models)
    classifier.main(["--articles", str(articles)])
    mapping = json.loads(mapping_path.read_text())
    assert mapping["Reviewed type"] == reviewed
    assert mapping["Shirt"]["category_id"] == SHIRTS.id
    classifier.main(["--articles", str(articles)])
    assert json.loads(mapping_path.read_text()) == mapping


def test_main_leaves_rejected_candidate_unmapped_and_samples_across_rows(tmp_path, monkeypatch):
    mapping_path = tmp_path / "mapping.json"
    articles = tmp_path / "articles.csv"
    pd.DataFrame(
        {
            "product_type_name": ["Ambiguous"] * 10,
            "product_group_name": ["Upper"] * 10,
            "prod_name": [f"Product {i}" for i in range(10)],
            "detail_desc": [""] * 10,
        }
    ).to_csv(articles, index=False)
    monkeypatch.setattr(classifier, "CATEGORY_MAPPING_PATH", mapping_path)
    monkeypatch.setattr(classifier, "load_taxonomy", lambda: [SHIRTS])
    monkeypatch.setattr(classifier, "get_settings", lambda: SimpleNamespace(clip_model_name="fake"))
    encoder = SimpleNamespace(encode_texts=lambda texts: np.ones((len(texts), 2)))
    monkeypatch.setattr(classifier, "ClipEncoder", lambda _: encoder)
    monkeypatch.setattr(classifier, "build_chat_model", lambda _: FakeChatModel(0, "high"))
    actual_classify = classifier.classify_with_llm

    def check_examples(model, product_type, product_group, examples, candidates):
        assert len(examples) == classifier.EXAMPLES_PER_TYPE
        assert examples[0].startswith("Product 0:")
        assert examples[-1].startswith("Product 9:")
        return actual_classify(model, product_type, product_group, examples, candidates)

    monkeypatch.setattr(classifier, "classify_with_llm", check_examples)
    classifier.main(["--articles", str(articles)])
    assert not mapping_path.exists()
