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
- MIT licence; history rewritten without commit trailers and without the old Amazon data.

## Evaluation (seven-day holdout, 819 customers)

New-item MAP@12: hybrid 0.0108, collaborative 0.0088, graph 0.0077, content-based 0.0054,
popularity 0.0050. The hybrid pairing was chosen on this holdout (see README).

Content-based now uses one taste vector per product group, searched within the sections
that group was bought from. The single averaged vector had turned a dress buyer's history
into leggings and trousers, including menswear.

## Demo examples

Ten customers, two per section, chosen by the documented criteria in `demo.py` (coherent
history, a similar shopper, co-purchase complements, a hybrid list within their sections).
Choosing them takes about 12 s and now happens at API startup.

## Open items

- The shopping assistant was not exercised: no LLM is configured in this environment, so
  there is no current chat screenshot. Configure `.env` and run `take_screenshots` without
  `--skip-chat`.
- The Docker image was not rebuilt after these changes.
- A second sample or holdout week would show whether the hybrid's lead is stable.
