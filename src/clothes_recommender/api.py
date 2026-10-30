"""HTTP API exposing search, recommendations and the shopping assistant.

Run with: uv run uvicorn clothes_recommender.api:create_app --factory
"""

import io
import json
import logging
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import date
from importlib.metadata import version
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, HTTPException, Query, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from clothes_recommender.chat.assistant import AssistantReply, ChatTurn
from clothes_recommender.chat.cocktails import CocktailReply
from clothes_recommender.cocktails import Cocktail, CocktailSourceError
from clothes_recommender.config import get_settings
from clothes_recommender.demo import DemoCustomer
from clothes_recommender.domain import Product, ProductFilter, ScoredProduct, User
from clothes_recommender.recommenders.base import Recommender, UnknownUserError
from clothes_recommender.services import Services, build_services
from clothes_recommender.vector_store import IndexNotBuiltError

logger = logging.getLogger(__name__)

MAX_TOP_K = 50
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
# Bounds decode memory: a small compressed file can still expand to gigabytes of pixels.
MAX_IMAGE_PIXELS = 25_000_000
MAX_CHAT_HISTORY = 50

TopK = Annotated[int, Query(ge=1, le=MAX_TOP_K)]


class ProductHit(BaseModel):
    product: Product
    score: float
    reasons: list[str] = Field(default_factory=list)


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)
    section: str | None = Field(default=None, description="Only this section, e.g. Menswear.")


class QueryPhoto(BaseModel):
    """A photo of an H&M garment outside the catalog, to try image search with."""

    photo_id: int
    name: str
    product_type: str
    colour: str


class PurchasedProduct(BaseModel):
    purchase_date: date
    product: Product


class UserProfile(BaseModel):
    user: User
    purchases: list[PurchasedProduct]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default=[], max_length=MAX_CHAT_HISTORY)


class CocktailSearchRequest(BaseModel):
    query: str = Field(default="", max_length=100)
    ingredient: str = Field(default="", max_length=100)
    non_alcoholic: bool = False


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    indexed_products: int
    catalog_products: int
    cocktails_enabled: bool = False
    chat_enabled: bool
    ranking_enabled: bool


def get_services(request: Request) -> Services:
    return request.app.state.services


ServicesDep = Annotated[Services, Depends(get_services)]


def _to_hits(services: Services, scored: Sequence[ScoredProduct]) -> list[ProductHit]:
    hits = []
    for item in scored:
        product = services.catalog.get(item.product_id)
        if product is not None:
            hits.append(ProductHit(product=product, score=item.score))
    return hits


def _read_image(upload: UploadFile) -> Image.Image:
    if upload.size is None or upload.size > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image must be at most 10 MB.")
    try:
        image = Image.open(io.BytesIO(upload.file.read()))
        if image.width * image.height > MAX_IMAGE_PIXELS:
            raise HTTPException(status_code=413, detail="Image resolution is too large.")
        image.load()  # Image.open is lazy; decode now so corrupt files fail here.
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as error:
        raise HTTPException(status_code=400, detail="Upload a valid PNG or JPEG image.") from error
    return image


def create_app(services: Services | None = None) -> FastAPI:
    """Build the app. Tests pass ready-made `services`; production builds them at startup."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if services is not None:
            app.state.services = services
            yield
            return
        app.state.services = build_services(get_settings())
        # Choosing the examples runs the recommenders for many customers (~10 s on the default
        # sample); do it before serving so the UI's first page load stays fast.
        _ = app.state.services.demo_customers
        yield
        app.state.services.store.close()

    app = FastAPI(
        title="Clothes Recommender API", version=version("clothes-recommender"), lifespan=lifespan
    )

    @app.exception_handler(IndexNotBuiltError)
    async def index_not_built(_: Request, error: IndexNotBuiltError) -> JSONResponse:
        return JSONResponse(status_code=503, content={"detail": str(error)})

    @app.exception_handler(UnknownUserError)
    async def unknown_user(_: Request, error: UnknownUserError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(error)})

    @app.get("/health")
    def health(services: ServicesDep) -> Health:
        indexed = services.store.count() if services.store.exists() else 0
        return Health(
            indexed_products=indexed,
            catalog_products=len(services.catalog),
            chat_enabled=services.assistant is not None,
            cocktails_enabled=services.cocktails is not None,
            ranking_enabled=services.learned is not None,
        )

    @app.get("/products/{product_id}")
    def get_product(product_id: int, services: ServicesDep) -> Product:
        product = services.catalog.get(product_id)
        if product is None:
            raise HTTPException(status_code=404, detail=f"Product {product_id} not found.")
        return product

    @app.get("/products/{product_id}/image", response_class=FileResponse)
    def get_product_image(product_id: int, services: ServicesDep) -> FileResponse:
        # product_id is an int, so the path cannot escape images_dir.
        path = services.images_dir / f"{product_id}.jpg"
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"No image for product {product_id}.")
        return FileResponse(path, media_type="image/jpeg")

    @app.get("/products/{product_id}/related")
    def related_products(
        product_id: int,
        services: ServicesDep,
        kind: Literal["complement", "alternative"] = "complement",
        top_k: TopK = 10,
    ) -> list[ProductHit]:
        product = get_product(product_id, services)
        hits = _to_hits(services, services.graph.related(product_id, kind, top_k))
        reason = (
            f"Bought on the same day by the same shoppers as {product.name}; "
            "different product group."
            if kind == "complement"
            else f"Same specific Shopify category, product type and section as {product.name}, "
            "with compatible mapped audience attributes and a higher approximate price."
        )
        for hit in hits:
            hit.reasons = [reason]
        return hits

    @app.get("/products/{product_id}/similar")
    def similar_products(
        product_id: int, services: ServicesDep, top_k: TopK = 10
    ) -> list[ProductHit]:
        """Products whose photos look most like this one's (CLIP item-to-item)."""
        get_product(product_id, services)
        similar = services.search.similar_to(product_id, top_k)
        if similar is None:
            raise HTTPException(
                status_code=404, detail=f"Product {product_id} has no photo in the image index."
            )
        return _to_hits(services, similar)

    @app.get("/products/{product_id}/graph")
    def product_graph(product_id: int, services: ServicesDep, top_k: TopK = 12) -> dict:
        get_product(product_id, services)
        return services.graph.neighborhood(product_id, top_k)

    @app.get("/sections")
    def sections(services: ServicesDep) -> list[str]:
        return services.catalog.distinct("section")

    @app.post("/search/text")
    def search_by_text(request: TextSearchRequest, services: ServicesDep) -> list[ProductHit]:
        product_filter = ProductFilter(sections=(request.section,) if request.section else ())
        return _to_hits(
            services, services.search.by_text(request.query, request.top_k, product_filter)
        )

    @app.get("/search/examples")
    def query_photos(services: ServicesDep) -> list[QueryPhoto]:
        """Example query photos; empty until scripts.fetch_query_photos has run."""
        if services.query_photos_dir is None:
            return []
        index = services.query_photos_dir / "query_photos.json"
        if not index.is_file():
            return []
        return [QueryPhoto.model_validate(row) for row in json.loads(index.read_text())]

    @app.get("/search/examples/{photo_id}/image", response_class=FileResponse)
    def query_photo_image(photo_id: int, services: ServicesDep) -> FileResponse:
        # photo_id is an int, so the path cannot escape query_photos_dir.
        path = services.query_photos_dir / f"{photo_id}.jpg" if services.query_photos_dir else None
        if path is None or not path.is_file():
            raise HTTPException(status_code=404, detail=f"No example photo {photo_id}.")
        return FileResponse(path, media_type="image/jpeg")

    @app.post("/search/image")
    def search_by_image(
        file: UploadFile, services: ServicesDep, top_k: TopK = 10
    ) -> list[ProductHit]:
        return _to_hits(services, services.search.by_image(_read_image(file), top_k))

    @app.get("/users")
    def list_users(services: ServicesDep) -> list[int]:
        return services.customers.user_ids()

    @app.get("/demo/customers")
    def demo_customers(services: ServicesDep) -> list[DemoCustomer]:
        return services.demo_customers

    @app.get("/users/{user_id}")
    def get_user_profile(user_id: int, services: ServicesDep) -> UserProfile:
        user = services.customers.get_user(user_id)
        if user is None:
            raise UnknownUserError(user_id)
        purchases = [
            PurchasedProduct(purchase_date=p.purchase_date, product=product)
            for p in services.customers.purchases_for(user_id)
            if (product := services.catalog.get(p.product_id)) is not None
        ]
        return UserProfile(user=user, purchases=purchases)

    @app.get("/users/{user_id}/recommendations")
    def recommend(
        user_id: int,
        services: ServicesDep,
        strategy: Literal["content", "collaborative", "graph", "repeat", "learned"] = "content",
        top_k: TopK = 10,
    ) -> list[ProductHit]:
        recommender: Recommender | None = {
            "content": services.content_based,
            "collaborative": services.collaborative,
            "graph": services.graph,
            "repeat": services.repeat,
            "learned": services.learned,
        }[strategy]
        if recommender is None:
            raise HTTPException(
                status_code=503, detail="Personalized ranking is not available yet."
            )
        hits = _to_hits(services, recommender.recommend(user_id, top_k))
        explainer = {
            "graph": services.graph,
            "collaborative": services.collaborative,
            "repeat": services.repeat,
            "learned": services.learned,
        }.get(strategy)
        if explainer is not None:
            for hit in hits:
                hit.reasons = explainer.reasons(user_id, hit.product.product_id)
        return hits

    @app.get("/evaluation")
    def evaluation(services: ServicesDep) -> dict:
        """Offline evaluation results written by `scripts.evaluate --output`."""
        path = services.evaluation_path
        if path is None or not path.is_file():
            raise HTTPException(
                status_code=404,
                detail="No evaluation results yet: run "
                "`uv run python -m scripts.evaluate --qdrant-path .qdrant --output "
                "data/evaluation.json`.",
            )
        return json.loads(path.read_text())

    @app.post("/chat")
    def chat(request: ChatRequest, services: ServicesDep) -> AssistantReply:
        if services.assistant is None:
            raise HTTPException(
                status_code=503,
                detail="Chat is disabled: set LLM_MODEL (and LLM_API_KEY / LLM_BASE_URL) in .env.",
            )
        return agent_reply(services.assistant, request)

    @app.post("/cocktails/search")
    def cocktail_search(request: CocktailSearchRequest, services: ServicesDep) -> list[Cocktail]:
        if services.cocktails is None:
            raise HTTPException(
                status_code=503, detail="Cocktail discovery is currently unavailable."
            )
        try:
            return services.cocktails.search(
                request.query, ingredient=request.ingredient, non_alcoholic=request.non_alcoholic
            )
        except CocktailSourceError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error

    @app.post("/cocktails/chat")
    def cocktail_chat(request: ChatRequest, services: ServicesDep) -> CocktailReply:
        if services.cocktail_assistant is None:
            raise HTTPException(
                status_code=503, detail="The cocktail assistant is currently unavailable."
            )
        return agent_reply(services.cocktail_assistant, request)

    def agent_reply(agent, request: ChatRequest):
        try:
            return agent.reply(request.message, request.history)
        except CocktailSourceError as error:
            raise HTTPException(status_code=503, detail=str(error)) from error
        except IndexNotBuiltError:
            raise
        except Exception as error:
            if getattr(error, "status_code", None) == 429:
                logger.warning("Shopping assistant provider is rate limited")
                raise HTTPException(
                    status_code=429,
                    detail="The assistant is busy. Please try again shortly.",
                ) from error
            # Provider SDKs raise their own exception types (auth, rate limit, network). Log the
            # details server-side only: upstream messages can contain URLs or request ids.
            logger.exception("Shopping assistant failed")
            raise HTTPException(
                status_code=502, detail="LLM provider request failed; see server logs."
            ) from error

    return app
