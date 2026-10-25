# Engineering workflow

The project is a portfolio demo for applied AI engineering. Changes should remain small,
reviewable and reproducible: KISS, DRY and YAGNI guide implementation choices.

## Team structure

Herdr provides visible worker terminals. A coordinator owns shared interfaces, API/service
wiring and integration. Workers share this checkout with explicit file ownership; no two
workers edit the same files concurrently. Existing uncommitted work was backed up before
the first edits. Workers do not commit or push.

| Workstream | Ownership | Evidence expected |
|---|---|---|
| H&M preparation | Preparation script and its tests | Basket counting, lift, empty data, fixtures and metadata consistency |
| Shopify enrichment | Classifier, saved mapping, enrichment and their tests | Valid taxonomy paths/values, conservative mapping, offline boundaries |
| CLIP batching | Encoder and its tests | Input order, chunk boundaries, normalization and cached-model equivalence |
| Graph baseline | Graph recommender and its tests | Ranking, evidence, exclusions, fallbacks, relations and bounded neighborhoods |
| Evaluation | Evaluation script and its tests | Hand-calculated AP, chronological split and no future purchase signals |
| Graph UI | Streamlit UI and its tests | Customer/product exploration, readable evidence and empty/error states |

The three initial workers were reassigned after handing off their changes. A completed
worker then reviewed graph and API integration without editing it. The coordinator runs
the combined checks and resolves findings. This keeps implementation and review separate
without assigning an agent to every small task.

## Contracts agreed before parallel implementation

- The graph implements the existing `recommend(user_id, top_k)` contract.
- Graph product hits include textual evidence. Related products distinguish co-purchased
  complements from higher-priced alternatives.
- A neighborhood contains bounded, namespaced nodes and edges whose endpoints exist.
- Evaluation uses a global chronological cutoff and rebuilds all purchase-derived
  signals from training purchases. Static product metadata remains available.
- The UI talks only to the API and uses existing Streamlit components.

## Integration checks

The combined suite passed 241 tests, with the model integration test deselected by the
normal offline test configuration. Ruff lint, formatting and Git whitespace checks
passed. A separate cached-CLIP check matched chunked and unchunked vectors within
`1.64e-7`. The real-sample API returned graph hits with evidence and bounded neighborhoods.
Browser checks exercised the graph view at 1440px and 390px without page-level horizontal
overflow or browser errors.
Final captures in `docs/screenshots/knowledge-graph-*.png` show light and dark desktop,
dark narrow-screen layouts and the curated customer menu. Both customer selectors were
checked with ten examples and access to all 200 users in the local preview. The built
container also passed imports, curated-route registration and packaged-taxonomy checks
as its non-root app user with networking disabled.

Review caught two integration issues before handoff: newly required enrichment fields
broke older prepared catalogs, and repeated common features caused redundant graph scans.
Compatibility defaults preserve old catalogs. Shared feature contributions are aggregated
before expanding their postings, retaining the same mathematical score. Explanations show
at most three reasons, prioritizing co-purchase evidence. A separate read-only re-review
confirmed both fixes.

User review then exposed a dark-mode contrast bug and unrelated higher-priced suggestions
within H&M's broad “Other accessories” type. Graph labels now follow the viewer's theme.
Alternatives require a shared specific Shopify category as well as type and section, and
reject conflicting mapped audience attributes. Unclassified or broad categories abstain.

The UI defaults to ten data-derived customer examples with descriptive labels and useful
starting products. Selection favors available graph evidence and diversity, while an
all-customer toggle preserves access to the full dataset. Evaluation continues to use its
chronologically eligible cohort, not this curated presentation set.

The local Docker image builds with the taxonomy mapping included and local datasets
excluded from its build context. Full H&M photo preparation and image-index validation
remain separate from these code and graph checks.

## Data and validation limits

The recovered checkout initially had 69 passing tests and four preparation fixture errors.
The data-preparation work also corrected lift to include baskets outside the selected
catalog. Mapping review found valid taxonomy IDs attached to overly specific or incorrect
categories; ambiguous H&M types remain unmapped instead of receiving confident guesses.

A local development smoke sample selects 200 shoppers with at least five training-period
purchases, with seed 42, from July 1 through September 22, 2020. Its verified transaction
shard covers June 15 through September 22. It contains 2,207 purchases and 1,703 products;
21 shoppers have purchases in the September 16–22 holdout. This small cohort validates
execution and metric output; it cannot establish comparative recommendation quality.
The sample stays in temporary local storage and is not redistributed.

One downloaded older transaction shard has an invalid Parquet footer. It was left
untouched and excluded from the smoke run; the valid final shard fully covers that run's
window. H&M photos are not available locally. The full photo preparation, H&M image-index
build and refreshed product screenshots therefore remain pending. Old screenshots in the
README are explicitly marked as legacy data.

## Deliberately deferred

GNN training, Neo4j, authentication and learned upsell ranking need evidence that the
baseline requires them. The graph uses purchase history as customer context; selecting
a demo customer is not an account or authentication system. Higher approximate prices
are not evidence of quality, margin or conversion uplift.
