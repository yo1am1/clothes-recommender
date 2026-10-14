# Clothy: multimodal fashion recommender

Clothy is an end-to-end recommender for a fashion catalog. It combines CLIP vector search,
two classic recommender algorithms and an LLM shopping assistant:

| Feature | How it works |
|---|---|
| Text-to-image search | The query is embedded with CLIP's text encoder and compared with the CLIP image embeddings of every product in Qdrant. |
| Image search | An uploaded photo is embedded with CLIP's image encoder and compared the same way. |
| Content-based recommendations | Each user has a *taste vector*, the recency-weighted mean of the embeddings of the products they bought. It is searched within the user's favourite categories, excluding past purchases. |
| Collaborative recommendations | User-based collaborative filtering. Users are compared by cosine similarity on the sparse user × product purchase matrix, and each product is scored by the similarity-weighted votes of the nearest neighbours. |
| Shopping assistant (RAG) | The LLM turns a message into a structured `SearchIntent`: a visual query plus category, brand, colour, price and rating filters. Qdrant retrieves the matching products, and the answer cites only those products. |

Users without usable purchase history get a Bayesian-average popularity ranking, which also fills
collaborative lists that come up short.

## Screenshots

### Content-based recommendations

The taste vector built from this shopper's 45 purchases retrieves visually similar items, mostly leggings and slim-fit jeans.

![Content-based recommendations](docs/screenshots/recommendations-content.jpg)

### Collaborative recommendations

Products bought by the most similar shoppers. The demo shoppers and their purchases are synthetic, so similarity between neighbours is low (about 0.06).

![Collaborative recommendations](docs/screenshots/recommendations-collaborative.jpg)

### Text-to-image search

The query "red floral summer dress" is compared with product photos directly; product titles are not searched.

![Text-to-image search](docs/screenshots/text-search.jpg)

### Image search

The query photo is a dress that is not in the index. The closest match is a maxi dress from the same brand, followed by other floral dresses.

![Image search](docs/screenshots/image-search.jpg)

### Shopping assistant

The LLM extracts the intent (category, maximum price, minimum rating), Qdrant applies it as a metadata filter, and the answer cites only the retrieved products.

![Shopping assistant](docs/screenshots/shopping-assistant.jpg)

## Architecture

```mermaid
flowchart LR
    UI[Streamlit UI] -- HTTP --> API[FastAPI]
    API --> S[ProductSearch]
    API --> CB[ContentBasedRecommender]
    API --> CF[CollaborativeRecommender]
    API --> A[ShoppingAssistant]
    S --> E[CLIP encoder]
    S --> Q[(Qdrant)]
    CB --> Q
    CB & CF --> P[PopularityRecommender]
    A --> LLM[[Any LLM provider]]
    A --> S
    B[scripts/build_index.py] --> E & Q
```

```
src/clothes_recommender/
├── config.py          # Settings from env / .env (pydantic-settings)
├── domain.py          # Product, User, Purchase, ProductFilter, Encoder protocol
├── catalog.py         # Product catalog + customer data access
├── embeddings.py      # CLIP encoder (lazy-loaded)
├── vector_store.py    # Qdrant wrapper: upsert, filtered search, vector lookup
├── search.py          # Text and image search
├── recommenders/      # content_based, collaborative, popularity
├── chat/              # llm (provider-agnostic), intent, prompts, assistant
├── services.py        # Composition root
├── api.py             # FastAPI app
└── ui.py              # Streamlit client
scripts/build_index.py # Download images, embed with CLIP, upsert into Qdrant
scripts/take_screenshots.py # Re-capture the README screenshots with Playwright
```

## Quick start

Requires [uv](https://docs.astral.sh/uv/) and Python 3.13.

```bash
uv sync --all-extras
cp .env.example .env                                   # optional: configure an LLM for the chat tab

uv run python -m scripts.build_index --limit 3000      # demo index: ~2.5 min on a 20-core CPU
uv run uvicorn clothes_recommender.api:create_app --factory --reload   # http://localhost:8000/docs
uv run streamlit run src/clothes_recommender/ui.py     # http://localhost:8501
```

Without `--limit` the whole catalog is indexed. Qdrant runs embedded (stored in `.qdrant/`)
unless `QDRANT_URL` is set, so no external service is needed. Embedded Qdrant allows one
process at a time: build the index before starting the API (or point both at a Qdrant server).

### Docker

```bash
docker compose up --build
docker compose exec api python -m scripts.build_index --limit 3000
```

### LLM configuration

The assistant works with any LangChain chat provider, configured only through `.env`:

```bash
LLM_PROVIDER=openai                       # openai | anthropic | azure_openai | ...
LLM_MODEL=gpt-4.1-mini
LLM_API_KEY=sk-...
LLM_BASE_URL=                             # set for OpenAI-compatible gateways
```

Any OpenAI-compatible gateway (OmniRoute, OpenRouter, LiteLLM, Ollama, vLLM) works with
`LLM_PROVIDER=openai` plus `LLM_BASE_URL`. Without `LLM_MODEL` every feature except chat still works.

## API

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Index size and whether chat is enabled |
| GET | `/products/{id}` | Product details |
| POST | `/search/text` | `{"query": "red floral dress", "top_k": 10}` |
| POST | `/search/image` | Multipart image upload |
| GET | `/users`, `/users/{id}` | User ids; profile with purchase history |
| GET | `/users/{id}/recommendations?strategy=content\|collaborative` | Personalised recommendations |
| GET | `/users/{id}/similar` | Most similar shoppers |
| POST | `/chat` | `{"message": "...", "history": [...]}` |

Interactive docs at `http://localhost:8000/docs`.

## Development

```bash
uv run pytest                        # fast, offline unit tests (fake encoder, in-memory Qdrant, fake LLM)
uv run pytest -m integration        # tests that download the real CLIP model
uv run ruff check . && uv run ruff format --check .

# Re-capture README screenshots (API + UI running; uses the local Google Chrome)
uv run python -m scripts.take_screenshots --query-image path/to/garment.jpg
```
