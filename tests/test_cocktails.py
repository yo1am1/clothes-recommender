import httpx
import pytest

from clothes_recommender.cocktails import CocktailCatalog, CocktailSourceError


def recipe(drink_id=1, alcohol="Alcoholic"):
    return {
        "idDrink": str(drink_id),
        "strDrink": "Fixture Fizz",
        "strAlcoholic": alcohol,
        "strGlass": "Highball",
        "strIngredient1": " Gin ",
        "strMeasure1": " 30 ml ",
        "strIngredient3": "Soda water",
        "strMeasure3": None,
        "strInstructions": "Stir over ice.",
        "strDrinkThumb": "https://www.thecocktaildb.com/images/fixture.jpg",
    }


def source(monkeypatch, payload):
    calls = []

    def get(url, params, timeout):
        calls.append((url.rsplit("/", 1)[-1], params))
        value = payload(url, params) if callable(payload) else payload
        return httpx.Response(200, json=value, request=httpx.Request("GET", url))

    monkeypatch.setattr("clothes_recommender.cocktails.httpx.get", get)
    return CocktailCatalog("fixture-key"), calls


def test_filter_summaries_are_enriched_and_blank_measures_are_retained(monkeypatch):
    catalog, calls = source(
        monkeypatch,
        lambda url, p: {
            "drinks": [recipe()] if "lookup" in url else [{"idDrink": "1", "strDrink": "Summary"}]
        },
    )
    first = catalog.search(ingredient="Gin")
    assert first[0].name == "Fixture Fizz"
    assert [(i.name, i.measure) for i in first[0].ingredients] == [
        ("Gin", "30 ml"),
        ("Soda water", None),
    ]
    assert first[0].source_url == "https://www.thecocktaildb.com/drink/1"
    catalog.get(1)
    assert calls == [("filter.php", {"i": "Gin"}), ("lookup.php", {"i": "1"})]


def test_non_alcoholic_filter_checks_full_recipe_status(monkeypatch):
    def payload(url, p):
        if "lookup" in url:
            return {
                "drinks": [recipe(int(p["i"]), "Non alcoholic" if p["i"] == "2" else "Alcoholic")]
            }
        return {"drinks": [{"idDrink": "1"}, {"idDrink": "2"}]}

    catalog, _ = source(monkeypatch, payload)
    assert [d.drink_id for d in catalog.search(non_alcoholic=True)] == [2]
    assert catalog.search("Fizz", ingredient="Almond") == []


@pytest.mark.parametrize("payload", [{}, {"drinks": "invalid"}, {"drinks": ["not a record"]}])
def test_bad_source_payload_is_unavailable_not_empty_success(monkeypatch, payload):
    catalog, _ = source(monkeypatch, payload)
    with pytest.raises(CocktailSourceError, match="unavailable"):
        catalog.search("Fizz")


def test_unknown_recipes_and_untrusted_images_are_handled(monkeypatch):
    catalog, _ = source(monkeypatch, {"drinks": None})
    assert catalog.get(1) is None
    assert catalog.search("unknown") == []
    row = recipe()
    row["strDrinkThumb"] = "http://localhost/private"
    catalog, _ = source(monkeypatch, {"drinks": [row]})
    assert catalog.get(1).image_url is None
    with pytest.raises(CocktailSourceError, match="verified"):
        catalog.get(99)


def test_source_error_does_not_expose_url_or_key(monkeypatch):
    def fail(*a, **kw):
        raise httpx.ConnectError("https://fixture-key/private")

    monkeypatch.setattr("clothes_recommender.cocktails.httpx.get", fail)
    with pytest.raises(CocktailSourceError) as caught:
        CocktailCatalog("fixture-key").search("Fizz")
    assert "fixture-key" not in str(caught.value)


def test_enrichment_is_bounded_and_no_search_is_network_free(monkeypatch):
    def payload(url, p):
        return (
            {"drinks": [recipe(int(p["i"]))]}
            if "lookup" in url
            else {"drinks": [{"idDrink": str(i)} for i in range(1, 100)]}
        )

    catalog, calls = source(monkeypatch, payload)
    assert catalog.search() == [] and not calls
    assert len(catalog.search("Fizz")) == 5
    assert len(calls) == 6
