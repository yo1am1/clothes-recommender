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
