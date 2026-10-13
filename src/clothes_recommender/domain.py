"""Core domain types shared by every layer (retrieval, recommenders, chat, API)."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

import numpy as np
from PIL.Image import Image
from pydantic import BaseModel


class Product(BaseModel):
    product_id: int
    name: str
    category: str
    brand: str
    color: str
    price: float
    rating: float
    review_count: int
    image_url: str
    product_url: str


class User(BaseModel):
    user_id: int
    name: str
    email: str
    city: str
    country: str
    gender: str


class Purchase(BaseModel):
    purchase_id: str
    user_id: int
    product_id: int
    purchase_date: date
    quantity: int


@dataclass(frozen=True)
class ProductFilter:
    """Structured constraints on product metadata.

    Fields combine with AND; values inside one field combine with OR.
    Empty tuples / None mean "no constraint".
    """

    categories: tuple[str, ...] = ()
    brands: tuple[str, ...] = ()
    colors: tuple[str, ...] = ()
    min_price: float | None = None
    max_price: float | None = None
    min_rating: float | None = None
    exclude_ids: tuple[int, ...] = ()

    def is_empty(self) -> bool:
        return self == ProductFilter()


@dataclass(frozen=True)
class ScoredProduct:
    product_id: int
    score: float


class Encoder(Protocol):
    """Maps images and texts into one shared, L2-normalised embedding space."""

    @property
    def dimension(self) -> int: ...

    def encode_images(self, images: Sequence[Image]) -> np.ndarray:
        """Return a float32 array of shape (len(images), dimension)."""
        ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return a float32 array of shape (len(texts), dimension)."""
        ...
