"""In-memory access to the product catalog, users and purchase history."""

from collections.abc import Iterable, Iterator
from pathlib import Path

import pandas as pd

from clothes_recommender.domain import Product, Purchase, User


class DataFileMissingError(FileNotFoundError):
    def __init__(self, path: Path) -> None:
        super().__init__(f"Data file not found: {path}. Run the app from the repository root.")


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

    def categories(self) -> list[str]:
        return sorted(self._df["category"].unique().tolist())


class CustomerData:
    """Users and their purchase history."""

    def __init__(self, users: pd.DataFrame, purchases: pd.DataFrame) -> None:
        self._users = users.set_index("user_id", drop=False)
        self._purchases = purchases.assign(
            purchase_date=pd.to_datetime(purchases["purchase_date"]).dt.date
        )

    @classmethod
    def from_csv(cls, users_path: Path, purchases_path: Path) -> "CustomerData":
        return cls(pd.read_csv(_require(users_path)), pd.read_csv(_require(purchases_path)))

    @property
    def purchases(self) -> pd.DataFrame:
        """Columns: purchase_id, user_id, product_id, purchase_date, quantity."""
        return self._purchases

    def user_ids(self) -> list[int]:
        return sorted(self._users.index.tolist())

    def get_user(self, user_id: int) -> User | None:
        if user_id not in self._users.index:
            return None
        return User.model_validate(self._users.loc[user_id].to_dict())

    def purchases_for(self, user_id: int) -> list[Purchase]:
        """Purchases of one user, most recent first."""
        rows = self._purchases[self._purchases["user_id"] == user_id]
        rows = rows.sort_values("purchase_date", ascending=False)
        return [Purchase.model_validate(r) for r in rows.to_dict(orient="records")]
