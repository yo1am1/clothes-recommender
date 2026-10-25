from datetime import date, timedelta

import pandas as pd
import pytest

from clothes_recommender.catalog import CustomerData, ProductCatalog
from clothes_recommender.demo import select_demo_customers
from clothes_recommender.domain import Product, ScoredProduct
from clothes_recommender.recommenders.graph import GraphRecommender
from clothes_recommender.recommenders.popularity import PopularityRecommender


def product(pid: int, section: str = "Ladieswear", group: str = "Garment Upper body") -> dict:
    return Product(
        product_id=pid,
        name=f"Product {pid}",
        product_type=f"Type {group}",
        product_group=group,
        section=section,
        colour="Black",
        description="",
        price=10,
    ).model_dump()


class FakeCollaborative:
    """Returns each customer's prepared list; score 0 marks popularity filler."""

    def __init__(self, lists: dict[int, list[int]], filler: dict[int, int]) -> None:
        self._lists = lists
        self._filler = filler
        self.calls: list[int] = []

    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        self.calls.append(user_id)
        ids = self._lists[user_id][:top_k]
        evidence = len(ids) - self._filler.get(user_id, 0)
        return [ScoredProduct(pid, 1.0 if i < evidence else 0.0) for i, pid in enumerate(ids)]


class Shop:
    """Customers with six purchases each, a recommendation pool per section, a linked focus."""

    def __init__(self) -> None:
        self.products: dict[int, dict] = {}
        self.purchases: list[tuple[int, int, date]] = []
        self.users: list[int] = []
        self.links: list[tuple[int, int, int, float]] = []
        self.recommended: dict[int, list[int]] = {}
        self.filler: dict[int, int] = {}
        self._next_id = 1000

    def _new(self, section: str, group: str) -> int:
        self._next_id += 1
        self.products[self._next_id] = product(self._next_id, section, group)
        return self._next_id

    def customer(
        self,
        user_id: int,
        sections: list[str],
        group: str = "Garment Upper body",
        linked: bool = True,
        recommend_section: str | None = None,
        recommend_group: str | None = None,
    ) -> list[int]:
        """`sections` lists the section of each purchase (oldest first)."""
        self.users.append(user_id)
        bought = [self._new(section, group) for section in sections]
        for day, pid in enumerate(bought):
            self.purchases.append((user_id, pid, date(2020, 9, 1) + timedelta(days=day)))
        if linked:
            complement = self._new(sections[-1], "Accessories")
            self.links.append((bought[-1], complement, 3, 2.0))
        self.recommended[user_id] = [
            self._new(recommend_section or sections[-1], recommend_group or group)
            for _ in range(10)
        ]
        return bought

    def select(self, limit: int = 10):
        catalog = ProductCatalog(pd.DataFrame(self.products.values()))
        customers = CustomerData(
            pd.DataFrame({"user_id": self.users, "age": 30}),
            pd.DataFrame(self.purchases, columns=["user_id", "product_id", "purchase_date"]),
        )
        graph = GraphRecommender(
            customers,
            catalog,
            PopularityRecommender(customers.purchases),
            pd.DataFrame(self.links, columns=["product_id", "other_id", "count", "lift"]),
        )
        self.collaborative = FakeCollaborative(self.recommended, self.filler)
        return select_demo_customers(customers, catalog, graph, self.collaborative, limit)


def ids(examples) -> list[int]:
    return [example.user_id for example in examples]


def test_coherent_customer_with_evidence_is_selected_with_a_readable_label():
    shop = Shop()
    bought = shop.customer(1, ["Menswear"] * 6)
    [example] = shop.select()
    assert example.user_id == 1
    assert example.label == "Menswear: Type Garment Upper body"
    assert example.focus_product_id == bought[-1]  # the purchase with complements
    assert "6 purchases of 6 products, mostly Menswear" in example.description


@pytest.mark.parametrize(
    ("setup", "reason"),
    [
        (lambda s: s.customer(1, ["Ladieswear"] * 5), "too few products"),
        (lambda s: s.customer(1, ["Ladieswear"] * 31), "too many products"),
        (lambda s: s.customer(1, ["Ladieswear"] * 4 + ["Menswear"] * 2), "mixed sections"),
        (lambda s: s.customer(1, ["Ladieswear"] * 6, linked=False), "no complements"),
        (
            lambda s: s.customer(1, ["Ladieswear"] * 6, recommend_section="Menswear"),
            "recommends another section",
        ),
        (
            lambda s: s.customer(1, ["Ladieswear"] * 6, recommend_group="Shoes"),
            "misses the main group",
        ),
    ],
)
def test_customers_failing_a_criterion_are_skipped(setup, reason):
    shop = Shop()
    setup(shop)
    assert shop.select() == [], reason


def test_lists_with_popularity_filler_are_skipped():
    shop = Shop()
    shop.customer(1, ["Ladieswear"] * 6)
    shop.filler[1] = 1  # one of the ten has no co-purchase evidence
    assert shop.select() == []


def test_short_lists_are_skipped():
    shop = Shop()
    shop.customer(1, ["Ladieswear"] * 6)
    shop.recommended[1] = shop.recommended[1][:9]
    assert shop.select() == []


def test_mostly_one_section_is_coherent_enough():
    shop = Shop()
    # 5 of 6 Ladieswear = 83%; recommendations may use either section bought.
    shop.customer(1, ["Menswear"] + ["Ladieswear"] * 5)
    assert ids(shop.select()) == [1]


def test_sections_take_turns_and_limit_is_respected():
    shop = Shop()
    for user in (1, 2, 3):
        shop.customer(user, ["Ladieswear"] * 6)
    shop.customer(4, ["Menswear"] * 6)
    shop.customer(5, ["Baby/Children"] * 6)
    # Ladieswear has the most candidates, so it goes first in every round; the one-candidate
    # sections follow alphabetically.
    assert ids(shop.select()) == [1, 5, 4, 2, 3]
    assert ids(shop.select(limit=2)) == [1, 5]


def test_a_new_main_group_is_preferred_within_a_section():
    shop = Shop()
    shop.customer(1, ["Ladieswear"] * 7, group="Garment Upper body")
    shop.customer(2, ["Ladieswear"] * 6, group="Garment Upper body")
    shop.customer(3, ["Ladieswear"] * 6, group="Garment Full body")
    # 1 has the longest history; then 3 brings a new group before 2 repeats one.
    assert ids(shop.select()) == [1, 3, 2]


def test_selection_stops_checking_once_the_limit_is_reached():
    shop = Shop()
    for user in range(1, 30):
        shop.customer(user, ["Ladieswear"] * 6)
    assert len(shop.select(limit=3)) == 3
    assert len(shop.collaborative.calls) == 3


@pytest.mark.parametrize("limit", [0, -1])
def test_nonpositive_limit_is_empty(limit):
    shop = Shop()
    shop.customer(1, ["Ladieswear"] * 6)
    assert shop.select(limit) == []
