# Recommendation audit and MVP continuation

The October 25 branch improved discovery signals but excluded every previous
purchase. Its new-item MAP therefore did not answer whether personalization beat
the strong H&M baseline: recent unique purchases, followed by last-week best-sellers.

The API now exposes `strategy=repeat` with purchase-date explanations. Existing
content/collaborative/graph strategies preserve their discovery behavior. Evaluation
compares an unrestricted popularity baseline, its explicitly named new-item variant,
repeat recommendations and discovery methods on the same full holdout. It reports
MAP@12, customer hit rate and macro recall@12; repeat transactions count once in
truth. Novel-item scores remain diagnostic over the original ranked lists.

## Reproduced validation

Cached real H&M parquet data covers the full July 1 to September 22, 2020 window.
The existing preparation helpers, 12 weeks, 5,000 customers and seed 42 reproduce
14,300 products, 59,040 sampled purchases, 3,035,225 crowd purchases and 757,633
co-purchase links. These files and photos remain ignored local data.

| Holdout end | Customers | Buy-again MAP@12 | Collaborative MAP@12 | Buy-again hit rate | Buy-again recall@12 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2020-09-01 | 883 | 0.0254 | 0.0114 | 10.5% | 4.7% |
| 2020-09-08 | 915 | 0.0243 | 0.0156 | 10.6% | 5.6% |
| 2020-09-15 | 844 | 0.0370 | 0.0171 | 13.9% | 6.9% |

Buy-again wins MAP on each earlier validation week. This is a useful baseline,
not evidence of a learned model improvement. The fixed learned design was then measured on the September 16–22 holdout; its
labels were not used in fitting or parameter selection during this continuation. Reproduce each earlier week
with `scripts.evaluate --data-dir data/mvp --holdout-end <date> --output <file>`.
Every purchase-derived crowd signal is rebuilt strictly before that week's cutoff.

The prepared cohort and catalog were selected from the full window; all strategies
share that selection bias. These sample-level results are neither a population
estimate nor comparable competition leaderboard results. Prices are approximate;
the historical catalog does not establish present stock or purchasing ability.

## Product direction

Clothy becomes a small fashion-discovery MVP: personalized product lists, text or
photo discovery, product detail with related items, and a separate quality page.
Use one quiet editorial identity: warm neutral surfaces, ink text, restrained rust
accent, serif headings, portrait product photos and native labelled controls.
Mobile stacks product cards and keeps native navigation accessible. Render one
page at a time, preserve graph evidence and handle missing images, empty results,
unavailable services and a disabled assistant explicitly. No checkout, fabricated
stock, account system or unmeasured model claims are part of this work.

## Learned ranking

A fixed linear logistic model reranks up to 100 unique past purchases, 50
collaborative candidates and 50 best-sellers. Features cover repeat status,
frequency, global-calendar purchase recency, collaborative score/rank, popularity,
and product-type/section affinity. It uses installed NumPy/SciPy, L2 regularization
and bounded L-BFGS fitting. No LightGBM or new dependency is needed for this slice.
Scores are ranking signals; their probabilities are not calibrated or displayed.

Each earlier validation model trains on the three seven-day windows immediately
before its target week. Every candidate and crowd signal uses only purchases
before that training window. Future positive items are not injected into candidate
lists. A regression changes all later purchases and verifies identical fitted
weights for an explicit training end. Persisted artifacts are bounded JSON with
finite weights and training metadata; evaluation refuses models trained into its
holdout. Missing models keep other recommendation strategies available.

| Validation end | Learned MAP@12 | Buy-again MAP@12 | Learned hit rate | Learned recall@12 |
| --- | ---: | ---: | ---: | ---: |
| 2020-09-01 | 0.0286 | 0.0254 | 12.0% | 5.6% |
| 2020-09-08 | 0.0305 | 0.0243 | 11.9% | 6.3% |
| 2020-09-15 | 0.0433 | 0.0370 | 17.1% | 9.5% |

## Final held-out week

The final model trains through September 15 on 279,519 candidate rows, including
1,318 positives. One measurement covers September 16–22 and 819 eligible customers.
No parameter adjustment followed this result.

| Strategy | MAP@12 | Customer hit rate | Macro recall@12 |
| --- | ---: | ---: | ---: |
| learned | 0.0302 | 13.9% | 7.6% |
| repeat | 0.0248 | 11.1% | 6.0% |
| collaborative | 0.0160 | 7.3% | 3.1% |
| popularity | 0.0113 | 7.1% | 2.7% |
| graph | 0.0097 | 4.8% | 2.1% |
| content | 0.0072 | 4.2% | 1.8% |

The observed learned-vs-buy-again MAP difference is +0.00534.
Its 95% paired-customer bootstrap interval is [-0.00057, +0.01124]
(2,000 resamples, seed 42). It crosses zero: the single final week does not establish
superiority over buy-again. The three positive validation results and higher final
point estimate support further evaluation, without a claim of population uplift.
This interval excludes catalog/cohort selection and model-choice uncertainty.

The buy-again baseline is deterministic: equal purchase dates break by product ID.
It scores 0.0248 here, so the earlier session's 0.0233 is not used as the comparison.
All crowd signals are rebuilt before the cutoff. Visual-style results use only
3,537 of 14,300 images in the local index, not the complete image catalog;
they do not establish a content-model improvement. The full-window catalog/cohort
selection bias applies to every strategy, and competition scores are not comparable.

Collaborative scoring now multiplies sparse rows directly, avoiding a dense
history-by-catalog matrix per customer. Existing hand-calculated ranking tests pass.

Reproduce model training and the report after preparing data:

```sh
uv run python -m scripts.train_ranker --data-dir data --train-through 2020-09-15
uv run python -m scripts.evaluate --data-dir data --ranker data/ranker.json \
  --qdrant-path .qdrant --output data/evaluation.json
```

Data, photos, model files, indices and result JSON remain outside Git under ignored
local data paths. No LLM call is needed for training, recommendations or evaluation.


## Subsequent tuning decision

The [October 28 tuning round](recommendation-tuning.md) tested wider candidate
pools, additional history/sales features, tree objectives and product families.
The frozen challenger failed separate confirmation: MAP gained only 2.81% and
macro recall fell. It was not promoted; current production scores and model
remain unchanged. See the report for trial definitions, cohort limits and results.
