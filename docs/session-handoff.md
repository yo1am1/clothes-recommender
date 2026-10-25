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
  `/search/examples`, `/evaluation`, and `strategy=hybrid` with per-component rank reasons.
- `HybridRecommender`: reciprocal rank fusion (k=60) of collaborative and content-based.
- UI: photos on cards, text search with examples and section filter, image search with
  example photos, "Looks similar" in the graph explorer, and an Evaluation tab.
- `evaluate --output` saves the JSON the Evaluation tab reads; the hybrid is evaluated
  whenever an image index is given.
- Screenshots re-captured from the H&M sample; legacy and coming-soon captures removed.

## Evaluation (seven-day holdout, 819 customers)

New-item MAP@12: hybrid 0.0103, collaborative 0.0088, graph 0.0077, popularity 0.0050,
content-based 0.0046. The hybrid pairing was chosen on this holdout (see README).

## Open items

- The shopping assistant was not exercised: no LLM is configured in this environment, so
  there is no current chat screenshot. Configure `.env` and run `take_screenshots` without
  `--skip-chat`.
- The Docker image was not rebuilt after these changes.
- A second sample or holdout week would show whether the hybrid's lead is stable.
