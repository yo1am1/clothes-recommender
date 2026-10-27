# Assistant and reliability update — 2026-10-27

The current branch remains `feat/image-search-and-evaluation`. The system audit
and independent reviews used three visible Codex workers in Orca. Five verified
failures were repaired separately: rejected photos are validated before preview;
chat context stays within API limits during long conversations; specific two-level
bag classes support alternatives; startup rejects incompatible vector dimensions
and closes failed service construction; index resume refreshes filtering metadata
without re-encoding existing photos. Image/model changes and removed catalog IDs
still require explicit index recreation. No current model or evaluation was tuned
as part of these repairs.

**Ask Clothy** now opens a native floating chat panel on every page. Its fragment
keeps chat interactions separate from page work. Conversation persists on close
and navigation, New chat clears it, and compact product previews open the existing
viewer. The historical `/assistant` route remains available but is hidden from
navigation. Disabled and failed-provider states remain explicit.

The local preview at UI port 18511/API port 18710 now uses the repository's configured
OpenAI-compatible OmniRoute gateway; no credentials or provider/session state are
committed. Real queries returned white Menswear sneakers below EUR40, and a
follow-up narrowed them below EUR20. Desktop feed/panel readiness measured about
0.45/0.27 seconds in the warm local browser; these small checks are not a load test.
Replies took roughly four seconds, including both model calls and retrieval.

Validation: 340 offline tests passed, one integration test deselected; Ruff passed.
Chrome exercised real provider replies, follow-up constraints, navigation/close
persistence, product dialogs, keyboard Escape/focus and full-transcript layouts at
1368, 768, 390 and 320 px. The screenshot script includes panel/input bounds checks
for narrow layouts. Screen readers, software keyboards, other browser engines,
remote Qdrant and concurrent-load behavior were not exercised.

Recommendation tuning is next: baseline learned MAP@12 remains 0.03018, hit rate 13.9%,
macro recall 7.6%, with the earlier paired comparison interval crossing zero.
Candidate reach and reranking both need work. Use earlier rolling windows for
selection and disclose that the final September 16–22 result has already been
inspected. Keep data/models/index/evaluation artifacts outside Git.

The earlier handoff is retained below as history.

---

# MVP continuation — 2026-10-27

The current branch is `feat/image-search-and-evaluation`. Completed ranking changes
were published individually: `8ece635` repeat baseline and metrics, `e6758fc` sparse
collaborative scoring, `bbe0c27` learned ranking and measured evidence. The UI is
rebuilt as native Streamlit pages, with real discovery, product details and a
quality page. See [recommendation audit](recommendation-audit.md) for definitions,
validation windows, final metrics, uncertainty and reproduction commands.

Local prepared data is `data/mvp`: 5,000 users, 14,300 products, 59,040 sample
purchases, 3,035,225 crowd purchases and 757,633 links, reproduced from the cached
H&M shard. Its model trains through September 15; the September 16–22 result is
0.03018 MAP@12 / 13.9% hit rate / 7.6% recall. Deterministic buy-again reaches
0.02484 / 11.1% / 6.0%. The 95% paired MAP interval crosses zero, so the single
week does not establish superiority. No model parameters were adjusted afterward.

The local image index holds 3,537 products. Bulk exact-photo retrieval continues
separately; source photos, data, index, model and result JSON are ignored local
artifacts. New downloads do not refresh an index already opened by the API.
Stop the API before resuming `scripts.build_index`, then restart it. The existing
repository-level `.qdrant` index was preserved; this preview uses `data/mvp/qdrant`.
Query examples contain twelve out-of-catalog photos.

Preview: UI http://127.0.0.1:18511, API http://127.0.0.1:18710. The preview's assistant
is explicitly disabled; no live provider call ran. Cached CLIP handles text/photo
search. To reproduce the preview without the private temporary launcher, set
`DATA_DIR=data/mvp QDRANT_PATH=data/mvp/qdrant`, start the normal API and point the
UI's `API_URL` at it. Normal `.env` loading applies to those commands; configure
optional providers deliberately.

Browser verification exercised the learned feed, profile restoration, product
dialog, real description/photo retrieval, graph navigation, visible quality
uncertainty, disabled assistant, and 1440 px / 390 px layouts. No browser errors
or page-level mobile overflow were observed. The final offline suite passed 313 tests
with one real-model integration test deselected; Ruff passed. The updated screenshot
command completed against the real API/UI. Package inspection confirms the
stylesheet ships in the wheel. Native Herdr was unavailable, so no hidden
teammates were created. The Docker image rebuilt successfully; imports, bundled
stylesheet, non-root execution and data exclusion passed with networking disabled.
No live provider check ran.

The previous handoff is retained below as history.

---

# Session handoff — 2026-10-25

The image pause from 2026-10-21 is over: product photos, text and image search, visually
similar products and content-based recommendations work end to end on the default sample.

## Data used for validation

Prepared without Kaggle from the Hugging Face mirror's CSVs (`prepare_data --archive
<dir with the CSVs>`), defaults otherwise: 12 weeks, 5,000 customers, seed 42. Result:
14,300 products, 59,040 purchases, 757,633 co-purchase links. `fetch_photos --workers 8`
saved 14,280 photos (20 are missing in the mirror) in about 25 minutes, 368 MB after
downsizing. `build_index` embedded all 14,280 in about 5 minutes at 50 img/s on 4 cores.
`fetch_query_photos` saved 12 example photos of articles outside the sample.

## Added in this session

- `prepare_data` reads an unzipped directory; `fetch_photos` downsizes like extraction does.
- API: `/products/{id}/similar`, `/sections` plus a section filter on text search,
  `/search/examples`, `/evaluation`, and reasons on collaborative hits; `/users/{id}/similar`
  was removed with user-based collaborative filtering.
- UI: photos on cards, text search with examples and section filter, image search with
  example photos, "Looks similar" in the graph explorer, and an Evaluation tab.
- `evaluate --output` saves the JSON the Evaluation tab reads; content-based is evaluated
  whenever an image index is given.
- Screenshots re-captured from the H&M sample; legacy and coming-soon captures removed.
- MIT licence; history rewritten without commit trailers and without the old Amazon data.

## Recommendation audit (2026-10-25)

The recommenders learned from the 5,000 sampled customers only, 2% of the window's
purchases. `prepare_data` now also writes `shopper_purchases.parquet` (3.0M purchases of
catalog products by ~480k anonymous shoppers). Popularity counts every shopper over 7 days;
collaborative filtering is item-to-item over every shopper, with a 7-day half-life on the
customer's history and popularity damping 0.3; evaluation rebuilds all crowd signals from
purchases before the cutoff. The hybrid was removed: with item-to-item CF it added nothing
on validation, and fusing content hurt. Padding photos to a square for CLIP was tested and
rejected (content-based fell from 0.0052 to 0.0032).

Settings were chosen on validation weeks starting 08-26, 09-02 and 09-09, then measured
once on the 09-16 week. New-item MAP@12: collaborative 0.0168 (was 0.0088), popularity
0.0119 (was 0.0050), graph 0.0101 (was reported as 0.0077), content-based 0.0054.

## Demo examples

Ten customers, two per section, chosen by the documented criteria in `demo.py` (coherent
history, ten collaborative recommendations backed by co-purchases and within their
sections, co-purchase complements for the graph). Selection happens at API startup.

## Open items

- The shopping assistant was not exercised: no LLM is configured in this environment, so
  there is no current chat screenshot. Configure `.env` and run `take_screenshots` without
  `--skip-chat`.
- The Docker image was not rebuilt after these changes.
- The catalog is the set of products the sample bought, which flatters every strategy.
