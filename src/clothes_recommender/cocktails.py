"""Optional, source-backed cocktail discovery through the official TheCocktailDB API."""

from functools import lru_cache
from urllib.parse import quote, urlparse

import httpx
from pydantic import BaseModel, PositiveInt


class Ingredient(BaseModel):
    name: str
    measure: str | None = None


class Cocktail(BaseModel):
    drink_id: PositiveInt
    name: str
    alcohol_status: str
    glass: str
    ingredients: list[Ingredient]
    instructions: str
    image_url: str | None = None
    source_url: str

    @property
    def alcohol_free(self) -> bool:
        return self.alcohol_status.casefold() == "non alcoholic"


class CocktailSourceError(RuntimeError):
    pass


class CocktailCatalog:
    def __init__(self, api_key: str) -> None:
        self._recipe = lru_cache(maxsize=128)(self._lookup)
        self._base = f"https://www.thecocktaildb.com/api/json/v1/{quote(api_key, safe='')}/"

    def _get(self, endpoint: str, **params: str) -> list[dict]:
        try:
            response = httpx.get(self._base + endpoint, params=params, timeout=5)
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict) or "drinks" not in payload:
                raise ValueError("Unexpected cocktail source response")
            drinks = payload["drinks"]
            if drinks is None or drinks == "no data found":
                return []
            if not isinstance(drinks, list) or not all(isinstance(d, dict) for d in drinks):
                raise ValueError("Unexpected cocktail source response")
            return drinks
        except (httpx.HTTPError, ValueError, AttributeError) as error:
            raise CocktailSourceError(
                "Cocktail discovery is unavailable. Please try again."
            ) from error

    def get(self, drink_id: int) -> Cocktail | None:
        return self._recipe(drink_id)

    def _lookup(self, drink_id: int) -> Cocktail | None:
        if drink_id <= 0:
            return None
        rows = self._get("lookup.php", i=str(drink_id))
        if not rows:
            return None
        row = rows[0]
        try:
            if int(row["idDrink"]) != drink_id:
                raise ValueError("Source returned a different recipe")
            image = row.get("strDrinkThumb") or ""
            parsed = urlparse(image)
            if parsed.scheme != "https" or parsed.hostname != "www.thecocktaildb.com":
                image = None
            return Cocktail(
                drink_id=drink_id,
                name=row["strDrink"],
                alcohol_status=row.get("strAlcoholic") or "Not specified",
                glass=row.get("strGlass") or "Not specified",
                ingredients=[
                    Ingredient(
                        name=name.strip(), measure=(row.get(f"strMeasure{i}") or "").strip() or None
                    )
                    for i in range(1, 16)
                    if (name := row.get(f"strIngredient{i}")) and name.strip()
                ],
                instructions=row.get("strInstructions") or "",
                image_url=image or None,
                source_url=f"https://www.thecocktaildb.com/drink/{drink_id}",
            )
        except (KeyError, TypeError, ValueError, AttributeError) as error:
            raise CocktailSourceError(
                "The recipe could not be verified. Try another drink."
            ) from error

    def search(
        self, query: str = "", *, ingredient: str = "", non_alcoholic: bool = False
    ) -> list[Cocktail]:
        if query.strip():
            rows = self._get("search.php", s=query.strip())
        elif ingredient.strip():
            rows = self._get("filter.php", i=ingredient.strip())
        elif non_alcoholic:
            rows = self._get("filter.php", a="Non_Alcoholic")
        else:
            return []
        results = []
        seen = set()
        # Bound enrichment requests even when filters omit hundreds of recipes.
        for row in rows[:10]:
            try:
                drink_id = int(row["idDrink"])
            except (KeyError, TypeError, ValueError) as error:
                raise CocktailSourceError("The recipe list could not be verified.") from error
            if drink_id in seen:
                continue
            seen.add(drink_id)
            drink = self.get(drink_id)
            if drink is None or (non_alcoholic and not drink.alcohol_free):
                continue
            if ingredient and not any(
                ingredient.strip().casefold() in i.name.casefold() for i in drink.ingredients
            ):
                continue
            results.append(drink)
            if len(results) == 5:
                break
        return results
