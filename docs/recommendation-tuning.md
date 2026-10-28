# Recommendation tuning — October 28, 2026

The current recommender is retained. A richer tree model improved the two
selection weeks but failed the frozen confirmation requirements. No experimental
model, feature change or LightGBM dependency was promoted to the application.
The quality problem remains open; this round does not establish an improvement.

## Protocol

Selection used all 883 and 915 eligible customers in weeks ending September 1
and 8, 2020. Each model trained on the three preceding seven-day windows.
Candidate generation, popularity, co-purchases and personal history stopped
strictly before each label week. Candidates were generated before attaching
labels; missing targets were never injected into training pools.

Every strategy used the same eligible customers and unique purchased-product
truth, including repeats. MAP@12 and macro recall average customers; hit rate
counts customers with at least one correct item. Candidate micro reach counts
reachable customer/product pairs over all truth pairs. It is not catalog coverage.

The prepared 5,000-customer cohort and 14,300-product catalog were selected from
the full observation window. That shared selection bias remains. These results
are not population estimates or competition leaderboard comparisons.

## Selection results

| Approach | Sep 1 MAP@12 | Sep 8 MAP@12 |
| --- | ---: | ---: |
| Buy-again baseline | 0.025383 | 0.024340 |
| Current linear ranker, freshly fitted | 0.028577 | 0.030524 |
| Wider CF/global candidates, linear ranker | 0.027724 | 0.032313 |
| Five additional history/interaction features, original pool | 0.029503 | 0.032111 |
| Binary tree, 15 features, wider pool | 0.028862 | 0.031354 |
| LambdaRank, 15 features, original pool | 0.027984 | 0.028475 |
| Rich binary tree, 24 features, wider pool | 0.030264 | 0.034084 |
| Exact-name/type/section families, 18 features | 0.030606 | 0.033940 |

The original pool contains up to 100 prior purchases, 50 collaborative candidates
and 50 global best-sellers. Increasing the latter limits to 150 and 100 raises
candidate micro reach from 13.32%/13.74% to 20.09%/21.70%, with roughly 245
candidates per customer. That gain alone does not reliably improve top-12 MAP.
A separate smaller expansion with personalized type/section best-sellers also
failed to improve MAP in both weeks.

The family channel added 23 reachable targets in each week but no correct
predictions from its added items. Its mean MAP nearly tied the rich model, with
lower hit rate and recall in both weeks. It was not selected. A fixed 30-day
buy-again cutoff was also inconsistent; no threshold grid was run.

The frozen rich design uses the wider pool and the original ten features, plus
reciprocal prior-purchase rank, recency-weighted frequency, repeat/popularity and
two collaborative/affinity interactions. Nine further fields describe past
one-day and 28-day sales, seven-day sales momentum, last observed sale age,
observed product tenure, user history age/size, colour affinity and group affinity.
Observed tenure is not a claimed product launch date. No future prices or sales
enter these fields.

The fixed experimental binary LightGBM 4.7.0 model used 100 rounds, 15 leaves,
maximum depth 5, learning rate 0.05, minimum leaf size 50, L2 penalty 1, seed 42
and deterministic two-thread execution. No selection-week early stopping was used.
The isolated experiment dependency was kept outside the project environment.

## Frozen confirmation

The rich design was frozen before measuring September 9–15. Both it and the
current comparator trained through September 8, on windows ending August 25,
September 1 and September 8. Predictions were saved before confirmation truth
was attached. All 844 eligible customers and 2,849 unique truth pairs remained.

| Approach | MAP@12 | Hit rate | Macro recall@12 | Recovered pairs |
| --- | ---: | ---: | ---: | ---: |
| Buy-again | 0.037037 | 13.86% | 6.86% | 148 |
| Current linear ranker | 0.043311 | 17.06% | 9.54% | 194 |
| Frozen rich challenger | 0.044529 | 17.30% | 9.14% | 199 |

Promotion required at least 5% relative MAP improvement over the freshly fitted
current model, with no overall hit-rate or macro-recall decrease. The challenger
achieved only 2.81% MAP improvement and reduced macro recall by 0.405 percentage
points. It failed both requirements and was rejected.

Its paired-customer MAP difference is +0.001218, with a 95% bootstrap interval
of [-0.003483, +0.006452] using 2,000 resamples and seed 42. The interval includes
zero. No predeclared section/history-age slice with at least 50 customers lost
more than 10% relative MAP; that does not override the overall failures.
Five extra recovered pairs alongside lower macro recall is valid: customers
receive equal weight in macro recall, while recovered-pair totals do not.

The three visible Orca workers completed candidate/feature experiments and
independent validation. Separate checks reproduced cached features, chronological
boundaries, model serialization, cohort denominators and metrics. The coordinator
independently recalculated confirmation metrics and the paired interval from saved
predictions. Per-week intervals exclude training, selection and population uncertainty.

The confirmation is now inspected and cannot serve as fresh evidence for another
design selected afterward. The previously inspected September 16–22 benchmark
was not rerun or used to select these changes. Its production values remain
0.03018 MAP@12, 13.9% hit rate and 7.6% macro recall.

## Operational checks

The OmniRoute assistant remains enabled. The application passed 342 offline tests,
with one real-model integration test deselected. A local 15-request check with
three simultaneous recommendation visitors returned twelve products every time:
p50 41 ms, p95 269 ms and maximum 275 ms. This is a bounded check of the retained
model, excluding photo delivery and LLM calls; it is not a load-test guarantee.

Future work needs materially different recommendation evidence and fresh
validation data. Repeatedly adjusting parameters against these inspected weeks
would not establish generalization. Raw per-product scores remain uncalibrated
ranking signals; their negative signs are not an arithmetic defect or a purchase
probability.
