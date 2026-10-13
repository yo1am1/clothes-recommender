FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.10 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"
WORKDIR /app

# Dependencies first so code changes don't invalidate the (large) torch layer.
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --extra ui --no-install-project

COPY README.md ./
COPY src ./src
COPY scripts ./scripts
COPY .streamlit ./.streamlit
RUN uv sync --locked --no-dev --extra ui

# Data is mounted at runtime (see docker-compose.yml), not baked into the image.
RUN useradd --create-home app
USER app

EXPOSE 8000 8501
CMD ["uvicorn", "clothes_recommender.api:create_app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
