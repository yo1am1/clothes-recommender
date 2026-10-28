# Fixed-pool provenance ablation — October 28, 2026

The application now supports an **opt-in feature version 2**, adding explicit repeat,
collaborative and popularity retrieval flags, plus reciprocal repeat/popularity
source ranks. Version 1 remains the default, with its original ten features.
The v2 vector has fifteen features; artifacts validate version and vector length,
and training, evaluation and service construction all propagate the version.

This isolates retrieval provenance: **candidate IDs, order, budget, original features,
training rows and positive counts are identical** between versions. Overlapping
sources retain their separate flags. A historical purchase outside the 100-item
repeat retrieval limit is not marked as retrieved by repeat merely because another
channel retrieves it. Existing CF score/rank features are retained.

## Exploratory results

These are previously inspected development windows, not new confirmation. Both
versions used the same three earlier labeled weeks and existing regularized linear
logistic fitting, with past-only histories/crowd signals. No wider retrieval or extra
metadata was included. All eligible users and complete unique-item truth were retained.

| Holdout end | Users | v1 MAP@12 | v2 MAP@12 | Hit rate v1 → v2 | Macro recall v1 → v2 |
| --- | ---: | ---: | ---: | --- | --- |
| 2020-09-01 | 883 | 0.028577 | 0.028034 | 12.00% → 12.23% | 5.61% → 5.61% |
| 2020-09-08 | 915 | 0.030524 | 0.030870 | 11.91% → 12.68% | 6.31% → 6.77% |

The paired MAP differences were −.000543 (95% interval −.002155 to +.000737)
and +.000346 (−.001108 to +.001967), using 2,000 paired customer bootstrap
resamples with seed 42. Both intervals include zero. This bootstrap does not account for training or
model-selection uncertainty. MAP did not improve consistently. **Do not promote this experiment.**
This narrow result does not test provenance combined with complementary retrieval,
a latent score, restored metadata or a nonlinear ranker.

Candidate reach stayed 433/3,251 pairs (13.32%) and 449/3,267 (13.74%). Repeat,
CF and popularity source reach was respectively 114/161/184 and 98/188/198 pairs;
these counts overlap and must not be summed as union reach. Provenance cannot by
itself retrieve previously missing articles.

Experimental artifacts and aggregate results were written only under
`/tmp/clothy-provenance`. SHA-256 checks verified retained products, users, sampled
purchases, crowd purchases and ranker files were unchanged. No live index, provider
configuration or serving model was changed. September 9–22 was not evaluated here.

## Reproduce independently

Use existing prepared data or an isolated preparation. These commands fit the
September 1 development comparison; repeat with training through September 1 and
holdout ending September 8 for the second window.

```sh
uv run python -m scripts.train_ranker --data-dir data/mvp --train-through 2020-08-25 --output /tmp/provenance-v1.json
uv run python -m scripts.train_ranker --data-dir data/mvp --train-through 2020-08-25 --feature-version 2 --output /tmp/provenance-v2.json
uv run python -m scripts.evaluate --data-dir data/mvp --holdout-end 2020-09-01 --ranker /tmp/provenance-v1.json
uv run python -m scripts.evaluate --data-dir data/mvp --holdout-end 2020-09-01 --ranker /tmp/provenance-v2.json
```

The v2 training CLI requires an explicit `--output` before reading data, preventing
an omitted output flag from overwriting the default ranker. Version 1 defaults
and older artifacts remain supported. Explicitly choosing the serving artifact's
path still overwrites it; use a separate path for all experiments.

Verification: 358 offline tests passed, one integration test deselected; Ruff lint
and format checks passed. Tests cover exact channel features, unchanged v1 feature
rows, truncated repeat retrieval, invalid artifact shapes, training chronology for
both versions, and training/evaluation/service-loading integration. No fresh
confirmation or online relevance/latency validation was performed.
