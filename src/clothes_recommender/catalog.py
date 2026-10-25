"""In-memory access to the product catalog, users and purchase history."""

import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from clothes_recommender.domain import Product, Purchase, User

logger = logging.getLogger(__name__)


class DataFileMissingError(FileNotFoundError):
    def __init__(self, path: Path) -> None:
        super().__init__(
            f"Data file not found: {path}. Prepare the dataset with "
            "`uv run python -m scripts.prepare_data` (see README)."
        )


def _require(path: Path) -> Path:
    if not path.exists():
        raise DataFileMissingError(path)
    return path


class ProductCatalog:
    def __init__(self, products: pd.DataFrame) -> None:
        self._df = products.set_index("product_id", drop=False)

    @classmethod
    def from_parquet(cls, path: Path) -> "ProductCatalog":
        return cls(pd.read_parquet(_require(path)))

    def __len__(self) -> int:
        return len(self._df)

    def __iter__(self) -> Iterator[Product]:
        for record in self._df.to_dict(orient="records"):
            yield Product.model_validate(record)

    def get(self, product_id: int) -> Product | None:
        if product_id not in self._df.index:
            return None
        return Product.model_validate(self._df.loc[product_id].to_dict())

    def get_many(self, product_ids: Iterable[int]) -> list[Product]:
        """Return products in the given order, silently skipping unknown ids."""
        return [p for pid in product_ids if (p := self.get(pid)) is not None]

    def distinct(self, field: str) -> list[str]:
        """Sorted distinct values of a text field, e.g. every product type."""
        return sorted(self._df[field].unique().tolist())


class CustomerData:
    """Users and their purchase history."""

    def __init__(self, users: pd.DataFrame, purchases: pd.DataFrame) -> None:
        # Nullable columns (e.g. unknown age) must become None, not NaN, for pydantic.
        users = users.astype(object).where(users.notna(), None)
        self._users = users.set_index("user_id", drop=False)
        self._purchases = purchases.assign(
            purchase_date=pd.to_datetime(purchases["purchase_date"]).dt.date
        )
        # Grouped once here so profile and recommendation requests don't rescan every purchase.
        ordered = self._purchases.sort_values(["user_id", "purchase_date"], ascending=[True, False])
        self._purchases_by_user: dict[int, list[Purchase]] = {
            int(user_id): [Purchase.model_validate(r) for r in group.to_dict(orient="records")]
            for user_id, group in ordered.groupby("user_id")
        }

    @classmethod
    def from_parquet(cls, users_path: Path, purchases_path: Path) -> "CustomerData":
        return cls(pd.read_parquet(_require(users_path)), pd.read_parquet(_require(purchases_path)))

    @property
    def purchases(self) -> pd.DataFrame:
        """Columns: user_id, product_id, purchase_date."""
        return self._purchases

    def user_ids(self) -> list[int]:
        return sorted(self._users.index.tolist())

    def get_user(self, user_id: int) -> User | None:
        if user_id not in self._users.index:
            return None
        return User.model_validate(self._users.loc[user_id].to_dict())

    def purchases_for(self, user_id: int) -> list[Purchase]:
        """Purchases of one user, most recent first."""
        return self._purchases_by_user.get(user_id, [])


def load_shopper_purchases(path: Path, customers: CustomerData) -> pd.DataFrame:
    """Every shopper's purchases (shopper_id, product_id, purchase_date).

    Data prepared before shopper_purchases.parquet existed falls back to the sampled
    customers' purchases, which makes popularity and collaborative filtering much weaker.
    """
    if path.exists():
        return pd.read_parquet(path)
    logger.warning("No %s: popularity and collaborative filtering use the sample only.", path)
    return customers.purchases.rename(columns={"user_id": "shopper_id"})[
        ["shopper_id", "product_id", "purchase_date"]
    ]
