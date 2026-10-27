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
not evidence of a learned model improvement. The final September 16–22 week is
reserved while implementation choices are being made. Reproduce each earlier week
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
