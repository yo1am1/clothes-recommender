"""The contract shared by every per-user recommender."""

from typing import Protocol

from clothes_recommender.domain import ScoredProduct


class UnknownUserError(LookupError):
    def __init__(self, user_id: int) -> None:
        super().__init__(f"Unknown user_id: {user_id}. Use an id from the users data file.")
        self.user_id = user_id


class Recommender(Protocol):
    def recommend(self, user_id: int, top_k: int = 10) -> list[ScoredProduct]:
        """Ranked products; each strategy defines whether past purchases are eligible."""
        ...
