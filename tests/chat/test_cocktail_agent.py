from types import SimpleNamespace

from langchain_core.messages import AIMessage

from clothes_recommender.chat.assistant import ChatTurn
from clothes_recommender.chat.cocktails import CocktailAssistant
from clothes_recommender.cocktails import Cocktail, Ingredient


def drink():
    return Cocktail(
        drink_id=2,
        name="Fixture Cooler",
        alcohol_status="Non alcoholic",
        glass="Highball",
        ingredients=[Ingredient(name="Lime", measure=None)],
        instructions="Stir.",
        source_url="https://www.thecocktaildb.com/drink/2",
    )


def test_cocktail_agent_uses_verified_recipes_and_exact_tool_call_ids(chat_model_replying):
    calls = []

    def search(query, **kw):
        calls.append((query, kw))
        return [drink()]

    catalog = SimpleNamespace(search=search, get=lambda i: drink() if i == 2 else None)
    model = chat_model_replying(
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "SearchCocktails",
                    "args": {"ingredient": "Lime", "non_alcoholic": True},
                    "id": "recipe_1",
                }
            ],
        ),
        "Try Fixture Cooler, an alcohol-free drink. Recipe: TheCocktailDB.",
    )
    response = CocktailAssistant(model, catalog).reply("An alcohol-free lime drink", [])
    assert response.cocktails == [drink()]
    assert calls == [("", {"ingredient": "Lime", "non_alcoholic": True})]


def test_recipe_followups_resolve_prior_ids_and_support_inspection(chat_model_replying):
    lookups = []

    def get(i):
        lookups.append(i)
        return drink() if i == 2 else None

    catalog = SimpleNamespace(get=get)
    model = chat_model_replying(
        AIMessage(
            content="",
            tool_calls=[
                {"name": "InspectCocktails", "args": {"drink_ids": [2, 999]}, "id": "recipe_2"}
            ],
        ),
        "The source does not state a lime quantity.",
    )
    response = CocktailAssistant(model, catalog).reply(
        "How much lime?", [ChatTurn(role="assistant", content="Try this", drink_ids=[2])]
    )
    assert lookups == [2, 2, 999]
    assert response.cocktails == [drink()]


def test_small_talk_makes_no_recipe_request(chat_model_replying):
    response = CocktailAssistant(chat_model_replying("You're welcome!"), SimpleNamespace()).reply(
        "Thanks", []
    )
    assert not response.cocktails and response.text == "You're welcome!"
