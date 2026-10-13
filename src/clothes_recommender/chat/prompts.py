"""Prompt templates for the shopping assistant."""

from collections.abc import Sequence

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from clothes_recommender.domain import Product

INTENT_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You turn a fashion shopper's latest message into a product search. "
            "Use the conversation for context (e.g. 'cheaper ones' refers to earlier items). "
            "Only set a constraint the shopper actually stated. "
            "Known categories: {categories}.",
        ),
        MessagesPlaceholder("history"),
        ("human", "{message}"),
    ]
)

ANSWER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are Clothy, a friendly and concise fashion shopping assistant. "
            "Recommend only products from the list below and never invent products, prices, "
            "ratings or links. Mention each recommended product's name, price, rating and "
            "link, formatted in markdown. If the list is empty but the shopper wanted "
            "products, say nothing matched and suggest how to broaden the search.\n\n"
            "Products:\n{products}",
        ),
        MessagesPlaceholder("history"),
        ("human", "{message}"),
    ]
)


def render_products(products: Sequence[Product]) -> str:
    """Render products as a compact numbered list for the answer prompt."""
    if not products:
        return "(none)"
    return "\n".join(
        f"{position}. [id {p.product_id}] {p.name} | brand: {p.brand} | ${p.price:.2f} | "
        f"rating {p.rating:.1f}/5 | {p.product_url}"
        for position, p in enumerate(products, start=1)
    )
