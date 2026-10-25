# Session handoff — 2026-10-21

Working tree contains the current work; nothing committed or pushed.

## Pause checkpoint

User requested image features be deferred. Both image-search tabs now show coming-soon messages with no search controls. Product cards show "Product images coming soon." and do not fetch or render photos, including cards in chat and graph results. Search API endpoints and downloaded assets remain available for future work. Do not re-enable images or image-search UI until requested.

Validation for this checkpoint: 31 UI tests passed and Ruff passed. Live browser checks passed; screenshots saved as docs/screenshots/recommendations-coming-soon.png, text-search-coming-soon.png, and image-search-coming-soon.png. Full-suite result below predates this small display change. Implementation is paused; photo indexing and end-to-end retrieval remain future work.

Preview: http://127.0.0.1:18501 (UI), http://127.0.0.1:18010 (API).
Data: /tmp/clothy-real-smoke-oggtyvdj. All 200 sampled customers retained; default selector shows ten curated examples.

Implemented: graph recommendations and evidence, relevant higher-price alternatives, bounded co-purchase weights (raw lift remains in evidence), ten curated examples, confirmation dialog for browsing all customers, clearer three-column cards and concise reasons, graph contrast fixes, evaluation script, taxonomy enrichment and photo downloader.

OmniRoute chat is now enabled in the running preview API using normal .env loading. GET /health reports chat_enabled=true; POST /chat returned HTTP 200 with a live greeting through OmniRoute. Restart command from the repo: DATA_DIR=/tmp/clothy-real-smoke-oggtyvdj QDRANT_PATH=/tmp/clothy-real-smoke-oggtyvdj/qdrant HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .venv/bin/uvicorn clothes_recommender.api:create_app --factory --host 127.0.0.1 --port 18010. Do not expose credentials. Preview Qdrant index is still empty; product retrieval requires photo indexing.

Photo downloader uses verified exact article IDs from the pinned public H&M mirror. Downloads may still be running; inspect /tmp/clothy-priority-photos.log and data-directory images before restarting. UI image cache now expires after 30 seconds. Finish downloading photos, build the image index, then verify product-search chat and screenshots of the latest cards. Existing screenshots in docs/screenshots verify graph contrast, not the newest card layout.

Final validation: 276 tests passed, one deselected; Ruff passed. Docker build passed earlier but was not rebuilt after the latest scoring/card/photo changes.

Existing Herdr workers clothy-data, clothy-taxonomy, clothy-clip were interrupted during wrap-up. Their recent output contains implementation notes. Preserve all pre-existing changes.
