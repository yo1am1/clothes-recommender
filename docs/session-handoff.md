# Agent tools and public repository — 2026-10-30

Model quality is hidden from navigation, with `/quality` and research records
preserved (`cb075b2`). Repository visibility was changed to public under Owner
instruction after a common credential-pattern/path scan of reachable history;
this is limited evidence, not a comprehensive confidentiality audit.

The assistant now has a bounded read-only SearchCatalog / InspectProducts loop.
Visual query refinement preserves extracted budget and metadata constraints.
The popup sends up to five prior card IDs; server-side catalog lookups resolve
comparisons and follow-ups. Invalid tools/args are returned as tool errors, and
exhaustion or empty output falls back to verified catalog items. No execution,
checkout or stock claims are supported. Existing chat responses remain compatible.

A real three-turn OmniRoute check returned five matches, compared two exact items,
and refined below€25 with matching price bounds. First turns took6.1/7.0seconds;
the third took56.7seconds because the SDK retried a429 after43seconds. Separate
fix `49f423c` disables SDK retries and returns a safe HTTP429 retry message.
Unit checks validate model configuration and response mapping; no deliberate
live rate-limit load test was run. Live evidence stays under ignored
`data/experiments/agent-smoke-20261030`. The next Owner direction is rebranding
as Curio, a broader agentic recommendation product, with cocktails optional.

---

# Latent and supervision-scale iteration — 2026-10-29

Offline BPR and deterministic recalled-negative sampling helpers plus bounded
comparison CLIs are delivered (`1426ac6`, `d698d2a`, `2a1c3a7`). The CLI dependency
preflight bug was fixed separately (`0b3628d`). Seven new designs were evaluated
on the same original Sep 1 / 8 buyers, pools and full truth. BPR/sampling were mixed
or worse; expanding supervision to 25,000 profiles raised Sep 8 MAP .033558→.036754
but lowered Sep 1 .031426→.027201. Five-week labels also failed consistency.
No design passed the frozen >=5% MAP improvement on both weeks plus v1 hit/recall
guards. No model promotion or new confirmation scoring occurred.

The new 20,000-shopper reservation (seed 20261029, target Sep 9–15) remains **unscored**.
Its digest is `75a63667f57ba62ec6d4e5a8b7b61eb3888100e334662853485ecd31c37d713e`.
Keep this fixed reservation available for one subsequently frozen design; do not
train on or resample it. When constructing another fresh customer cohort, exclude
these reserved IDs, the original 5,000, the spent 20,000 and the added training IDs.
Training-only bundle
`data/experiments/ranker-training-20261028` preserves original users and adds 20,000
selected strictly before Jul 22 (seed 20261030), excluding the 45,000 prior/reserved IDs.
Its labels stop Sep 1; do not assume it contains Sep 8 labels for later model fitting.

A machine restart erased the previous /tmp logs, optional libraries and caches.
The same reservation was restored and hash-verified; the unchanged protocol was
rerun with persistent ignored evidence under `data/experiments/iteration2-run`.
No private memberships, predictions, models, caches or dependencies are committed.
See [the iteration record](recommendation-iteration2.md), chart and aggregate JSON
for all arms, actual training counts, uncertainty, source audit and reproduction.
Serving datasets/model/index/provider/configuration remain unchanged; before/after
serving file hashes match. Fresh v1/repeat and previous challenger controls match
prior MAP to 1e-12. 526 offline tests pass, one integration test deselected;
Ruff lint/format pass. The same two native teammates completed helper/research and
independent review tasks before interruption; all tasks settled, conversations retained.
No online/load or prospective utility check was performed. Next candidate intervention:
distinct past-only retrieval, evaluated for incremental reach and final MAP.

---

# Ranking follow-up — 2026-10-28

Past-only context features are delivered (`954ce91`); serving does not use them.
A fixed three-arm comparison held candidate pools/truth constant. Binary loss
selection gave MAP .031426/.033558 versus the previous expanded .031352/.030757
and v1 .028577/.030524. Context binary .030283/.035159 and LambdaRank .027349/.031691
were inconsistent; no per-week winner switching.

Before selecting new target outcomes, froze the simpler 24-feature loss-selection
model/rules and sampled 20,000 past-eligible crowd shoppers excluding all original
5,000. On Sep9–15, 2,882 active buyers: v1 MAP .042484/hit16.59%/recall8.89%;
challenger .044104/17.94%/9.68%. MAP +3.81% failed the5% practical gate and paired
interval [−.000812,+.003840] includes zero. No major declared slice warning; no
promotion. This is customer-disjoint labels in a shared historical calendar/catalog,
not a new-time or completely independent dataset. All20,000 reserved memberships
are now inspected: never reuse them as fresh, nor sample future validation without
excluding them and the original cohort. Private reservation/crosswalk is at
`data/experiments/confirmation-20261028`.

See [the follow-up record](recommendation-ranking-followup.md) and aggregate JSON
for every arm, all slices, sampling/frozen rules and reproduction. Serving data,
model/index/provider/configuration unchanged. 446 offline tests pass, one integration
test deselected, Ruff/format pass. Only the same two visible teammates were used;
all implementation/review tasks settled with their conversations retained.

---

# Complementary retrieval implementation — 2026-10-28

Parquet shard preparation is delivered (`05a463b`). An isolated enriched bundle
reproduces all retained fields exactly and has a verified real user/crowd crosswalk.
One truncated raw shard was reconstructed only in an isolated source copy; all9
raw files match a pinned mirror SHA-256 manifest. Serving data/model remain unchanged.

Offline family/basket/SVD32-fold-in channels and the comparison harness are complete.
On Sep1/8, reachable truth pairs rose433→533 and449→559. Expanded-tree MAP was
.031352/.030757 versus v1 .028577/.030524, with higher hit/macro recall but both
paired MAP intervals crossing zero. The second gain is below5%; no model promotion
or independent confirmation was performed. See the [retrieval record](recommendation-retrieval.md)
and aggregate JSON for protocols, limitations and reproduction. Both development
weeks remain inspected; no fresh confirmation labels were consumed.

Only two teammates continued after the Owner limit; extra review was cancelled.
Implementation and integration-review tasks settled, conversations retained.

---

# Provenance experiment — 2026-10-28

Opt-in ranker feature v2 adds explicit source flags and repeat/popularity ranks.
V1 defaults/serving artifacts remain unchanged. The fixed-pool comparison on the
already-inspected September 1/8 development weeks showed mixed MAP, with paired
intervals crossing zero; **not promoted**. Candidate pools and base features were
identical, and retained input/model hashes were verified unchanged. See the
[provenance record](recommendation-provenance.md) for exact results and reproduction.
358 offline tests pass, one integration test deselected; Ruff/format pass.
The broader retrieval/latent-signal and judged-discovery work remains outstanding.

---

# Input-contract implementation — 2026-10-28

New preparations preserve product family/department/index, normalized per-event
prices and sales channels, and an explicit anonymous sampled-user/crowd crosswalk.
`--data-dir` allows isolated experiment output. Synthetic regression checks cover
identity mismatch, source-unit prices, invalid channels and serving compatibility;
349 offline tests pass, one integration test deselected, Ruff/format pass.
Serving datasets/models/indices were not regenerated. Old bundles still work;
new fields require a new preparation. See README Data for the field contract.

---

# Research continuation — 2026-10-28

Three visible Orca Codex workers completed primary-source recommendation research.
The [synthesis](recommendation-research.md) documents the independently reproduced
424/2,702 candidate reach, discarded metadata and missing sampled/crowd identity
crosswalk, original top-team retrieval evidence, and alternative dataset limits.
Keep/enrich H&M; prioritize retrieval provenance and input contracts, with separate
judged discovery evaluation. Limited failed trials do not establish impossibility.
No product/model/data/index/dependency changes were made. September 16–22 was
replayed read-only for diagnosis after tuning; it remains an already-inspected
retrospective benchmark. No September window is fresh for a newly selected design.

---

# Continuation — 2026-10-28

The OmniRoute popup is delivered on `feat/image-search-and-evaluation`. A later
fix (`10d30cc`) rechecks chat availability after backend restarts/configuration
changes instead of retaining stale page health. The full suite passes 342 tests
with one integration test deselected; source lint passes.

The three visible Orca Codex workers completed the bounded recommendation tuning
round and independent reviews. Wider retrieval improves candidate reach; extra
features and trees gave selection gains. The frozen rich24 design then failed
September 9–15 confirmation: MAP 0.044529 vs current 0.043311 (+2.81%, below 5%),
macro recall 9.14% vs 9.54%. It was rejected. No experimental model or dependency
was promoted, and production data/model/index/evaluation files remain unchanged.
See [the tuning record](recommendation-tuning.md) for definitions and results.
Do not reuse the now-inspected confirmation as a fresh test for another design.
September 16–22 was not rerun during this round.

The preview remains at UI port 18511/API port 18710 with configured OmniRoute chat enabled.
A bounded three-visitor recommendation check returned twelve products in every
request, with p95 269 ms; this excludes image and model-provider latency.
App-wide lag is not established as resolved by that limited check. All teammate
tasks settled and the original visible terminal conversations were retained.

---

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
