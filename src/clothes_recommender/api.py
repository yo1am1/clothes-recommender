"""HTTP API exposing search, recommendations and the shopping assistant.

Run with: uv run uvicorn clothes_recommender.api:create_app --factory
"""

import io
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
from clothes_recommender.config import get_settings
from clothes_recommender.domain import Product, ScoredProduct, User
from clothes_recommender.recommenders.base import Recommender, UnknownUserError
from clothes_recommender.recommenders.collaborative import SimilarUser
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


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=300)
    top_k: int = Field(default=10, ge=1, le=MAX_TOP_K)


class PurchasedProduct(BaseModel):
    purchase_date: date
    product: Product


class UserProfile(BaseModel):
    user: User
    purchases: list[PurchasedProduct]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[ChatTurn] = Field(default=[], max_length=MAX_CHAT_HISTORY)


class Health(BaseModel):
    status: Literal["ok"] = "ok"
    indexed_products: int
    chat_enabled: bool


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
        return Health(indexed_products=indexed, chat_enabled=services.assistant is not None)

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

    @app.post("/search/text")
    def search_by_text(request: TextSearchRequest, services: ServicesDep) -> list[ProductHit]:
        return _to_hits(services, services.search.by_text(request.query, request.top_k))

    @app.post("/search/image")
    def search_by_image(
        file: UploadFile, services: ServicesDep, top_k: TopK = 10
    ) -> list[ProductHit]:
        return _to_hits(services, services.search.by_image(_read_image(file), top_k))

    @app.get("/users")
    def list_users(services: ServicesDep) -> list[int]:
        return services.customers.user_ids()

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
        strategy: Literal["content", "collaborative"] = "content",
        top_k: TopK = 10,
    ) -> list[ProductHit]:
        recommender: Recommender = (
            services.content_based if strategy == "content" else services.collaborative
        )
        return _to_hits(services, recommender.recommend(user_id, top_k))

    @app.get("/users/{user_id}/similar")
    def similar_users(user_id: int, services: ServicesDep, top_k: TopK = 5) -> list[SimilarUser]:
        return services.collaborative.similar_users(user_id, top_k)

    @app.post("/chat")
    def chat(request: ChatRequest, services: ServicesDep) -> AssistantReply:
        if services.assistant is None:
            raise HTTPException(
                status_code=503,
                detail="Chat is disabled: set LLM_MODEL (and LLM_API_KEY / LLM_BASE_URL) in .env.",
            )
        try:
            return services.assistant.reply(request.message, request.history)
        except IndexNotBuiltError:
            raise
        except Exception as error:
            # Provider SDKs raise their own exception types (auth, rate limit, network). Log the
            # details server-side only: upstream messages can contain URLs or request ids.
            logger.exception("Shopping assistant failed")
            raise HTTPException(
                status_code=502, detail="LLM provider request failed; see server logs."
            ) from error

    return app
