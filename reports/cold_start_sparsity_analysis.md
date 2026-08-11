# Cold-Start and Sparsity Impact Analysis (Week 12)

## 1. Recap: why this dataset already excludes the most extreme cold-start cases

The train/test matrix used throughout this project has 92.63% sparsity (566 users × 1,286 movies, source: `reports/hyperparameter_tuning_results.csv` fit logs, e.g. `python scripts/analyze_cold_start.py` output for the full baseline). That sparsity figure already reflects a dataset that was deliberately filtered in Week 3: `filter_ratings()` in `src/data/preprocessor.py` applies an iterative `min_user_ratings=20, min_movie_ratings=20` threshold before the train/test split ever happens.

That means every user and movie in this dataset, by construction, already has **at least 20 ratings** somewhere in the full (pre-split) data. The genuinely hardest cold-start case — a brand-new user with zero or one rating, or a brand-new movie nobody has rated yet — was already excluded from the dataset before Week 1's modeling work even started. This week's simulation therefore cannot measure true cold-start behaviour; it can only measure how these models degrade as an *existing, moderately-active* user's visible history is artificially shrunk. See §6 for why this distinction matters.

## 2. Cold-start simulation results (Task 1)

Method: `scripts/analyze_cold_start.py` — for each truncation level, each user's training rows are randomly downsampled to at most `max_ratings` (via `numpy.random.default_rng(42)`, no replacement; users with fewer rows keep everything), the model is refit on the truncated data, and RMSE is measured against the full, untouched test set.

| Model | max_ratings | RMSE | MAE | n_predictions |
|---|---|---|---|---|
| UBCF | 1 | 1.2605 | 0.9167 | 5,974 |
| UBCF | 3 | 1.0495 | 0.8074 | 9,751 |
| UBCF | 5 | 1.0031 | 0.7765 | 11,385 |
| UBCF | 10 | 0.9531 | 0.7360 | 12,732 |
| UBCF | 20 | 0.9337 | 0.7248 | 13,266 |
| UBCF | full | 0.8430 | 0.6407 | 13,406 |
| IBCF | 1 | 1.2605 | 0.9167 | 5,974 |
| IBCF | 3 | 1.0571 | 0.8143 | 9,751 |
| IBCF | 5 | 1.0388 | 0.8016 | 11,385 |
| IBCF | 10 | 1.0705 | 0.8197 | 12,732 |
| IBCF | 20 | 1.0924 | 0.8409 | 13,266 |
| IBCF | full | 0.8769 | 0.6731 | 13,406 |

**Observed trend:** UBCF improves strictly monotonically at every truncation level (1.2605 → 1.0495 → 1.0031 → 0.9531 → 0.9337 → 0.8430) — more training data per user consistently helps. **IBCF does not follow a clean monotonic trend.** It improves from 1 to 5 ratings/user (1.2605 → 1.0571 → 1.0388), then *gets worse* at 10 and 20 ratings/user (1.0705, then 1.0924) — its worst point in the whole sweep after max_ratings=1 — before recovering sharply to 0.8769 at the full, untruncated dataset. At `max_ratings=1`, both models produce numerically identical RMSE (1.2605) — with exactly one rating per user, both effectively degrade to a low-information regime, which is a reasonable, if coincidental-looking, floor case rather than a bug (verified independently by matching `n_predictions=5,974` between them at that level).

The IBCF non-monotonicity is a genuine finding, not resolved in this report. See §4 for a discussion of a plausible mechanism.

## 3. Sparsity impact results (Task 2)

Method: `scripts/analyze_sparsity_impact.py` — every test-set prediction is bucketed by `n_neighbours`, the number of valid neighbours `_predict_with_support()` actually used for that specific prediction (0 for the fallback path), and RMSE is computed within each bucket.

| Model | Bucket (n_neighbours) | RMSE | n_predictions |
|---|---|---|---|
| UBCF | 0-2 | 1.1087 | 122 |
| UBCF | 3-5 | 0.9658 | 277 |
| UBCF | 6-10 | 0.9833 | 1,026 |
| UBCF | 11-20 | 0.8236 | 11,981 |
| IBCF | 11-20 | 1.0021 | 330 |
| IBCF | >20 | 0.8735 | 13,076 |

**Observed trend:** UBCF mostly improves as neighbour support increases (1.1087 → 0.9658 → 0.9833 → 0.8236) but is **not perfectly clean** — there's a small uptick between the 3-5 bucket (0.9658) and the 6-10 bucket (0.9833) before the sharp improvement at 11-20. **IBCF's bucket coverage doesn't actually exercise the low-support range at all**: 3 of the 4 requested buckets (0-2, 3-5, 6-10) contain zero predictions. IBCF's tuned hyperparameters (k=30, min_support=1) are permissive enough that in this dataset it almost never produces a prediction backed by fewer than 11 valid neighbours — 13,076 of its 13,406 total predictions (97.5%) fall into a `>20` overflow bucket outside the requested range entirely. Within the two buckets IBCF does populate, RMSE does improve (1.0021 → 0.8735), but that's a two-point comparison, not evidence across the sparsity range the buckets were designed to probe.

## 4. Discussion: which model degrades faster under cold-start?

**IBCF degrades faster and less predictably than UBCF as available data shrinks**, which is broadly consistent with the algorithmic difference between the two approaches, though the specific non-monotonic bump (§2) is a more precise and more concerning finding than "IBCF is just worse."

UBCF's neighbour-finding step depends on locating other *users* with overlapping taste. Truncating one user's ratings mostly affects predictions made *for* that user (fewer of their own data points to compare against neighbours), while the rest of the user-user similarity matrix stays comparatively well-populated (566 other users, most still contributing signal). This matches the clean, monotonic UBCF curve in §2.

IBCF's item-item similarity matrix, by contrast, is built from co-rating counts and mean-centred correlations *between movies*, computed across the whole truncated training set at once. Truncating every user's history simultaneously to a small fixed count (e.g. 10 or 20) doesn't just remove data — it changes *which* item pairs have enough shared raters to compute a similarity for at all, and at low-to-moderate truncation levels (10-20 ratings/user), it's plausible that a moderate number of newly-co-occurring item pairs pass IBCF's very permissive `min_support=1` filter with sparse, noisy support, which could inject weak/noisy similarities into `recommend()`/`predict()`'s top-k neighbour selection before there's enough co-rating volume for the noise to average out. That would explain why RMSE briefly *worsens* at 10-20 ratings/user rather than steadily improving. **This is a plausible mechanism, not something proven in this report** — a full explanation would require directly inspecting how the item-item similarity matrix's coverage and value distribution change between the 5, 10, and 20 truncation levels, which was not done here.

The sparsity-bucket results reinforce the same asymmetry from a different angle: UBCF, tuned with a high `min_support=10`, actually produces predictions across the full range of neighbour-support levels (from as few as 0-2 up to 20 neighbours), and those low-support predictions are measurably worse (RMSE 1.1087 in the 0-2 bucket vs 0.8236 in 11-20) — a real, quantified cold-start-like penalty. IBCF, tuned with `min_support=1`, essentially never operates in a low-support regime in this dataset at all, which means its Week 8-9-10 headline numbers say nothing about how it would behave if it ever did.

## 5. IBCF Non-Monotonicity Investigation

### 5.1 The observed anomaly

As reported in §2 and §4, IBCF's cold-start curve is not monotonic at its deployed configuration (k=30, min_support=1): RMSE improves from max_ratings=1 through 5 (1.2605 → 1.0571 → 1.0388), then **gets worse** at both max_ratings=10 (1.0705) and max_ratings=20 (1.0924) — a rise of +0.0536 over those two steps — before recovering sharply at the full, untruncated dataset (0.8769).

### 5.2 Hypothesis

IBCF's deployed `min_support=1` is extremely permissive: an item pair needs only a single shared rater to be treated as having a valid similarity. The hypothesis is that at moderate truncation levels (10-20 ratings/user), enough new item pairs cross that trivial threshold with thin, noisy support to actively hurt `predict()`'s top-k neighbour selection — before the full dataset's much higher co-rating volume dilutes that noise back out.

### 5.3 Controlled comparison

`scripts/analyze_cold_start.py` was extended to re-run the same truncation sweep for IBCF at k=20/min_support=3 — Week 8's actual RMSE-optimal IBCF configuration (`reports/hyperparameter_tuning_results.csv`, model=IBCF, min RMSE row: k=20, min_support=3, rmse=0.8729), not min_support=5 as originally guessed when this investigation was requested. Results saved to `reports/cold_start_ibcf_min_support3_results.csv`.

| max_ratings | IBCF (min_support=1, deployed) | IBCF (min_support=3, RMSE-optimal) |
|---|---|---|
| 1 | 1.2605 | 1.2605 |
| 3 | 1.0571 | 1.0495 |
| 5 | 1.0388 | 1.0045 |
| 10 | 1.0705 | 0.9673 |
| 20 | 1.0924 | 0.9901 |
| full | 0.8769 | 0.8729 |

### 5.4 Conclusion

**Partially confirmed.** The non-monotonicity shrinks substantially but does not fully disappear when min_support is raised from 1 to 3:

- At min_support=1, RMSE worsens over **two consecutive steps** (5→10→20), a total rise of **+0.0536**.
- At min_support=3, RMSE worsens over **one step only** (10→20: 0.9673 → 0.9901), a rise of **+0.0228** — roughly 43% the magnitude of the min_support=1 bump. The 5→10 transition, which broke at min_support=1, correctly improves at min_support=3 (1.0045 → 0.9673).

This is real, evidenced support for the hypothesis — more required co-rating evidence measurably reduces the anomaly — but it is not complete proof that min_support=1 is the *sole* cause, since a smaller residual bump still exists at min_support=3. Something beyond min_support alone (possibly interacting with the k=20 vs k=30 difference between the two configs, which was not isolated separately in this comparison) likely also contributes.

### 5.5 Connection to the Week 8 tuning tradeoff

Week 8's tuning report already flagged that IBCF's Precision@10-optimal configuration (k=30, min_support=1) was chosen over its RMSE-optimal configuration (k=20, min_support=3) specifically because ranking quality was prioritized for a recommender system, on a "thin statistical basis" of very low absolute co-rating requirements. This week's finding is a **concrete, realized consequence of that exact tradeoff**, not a new, unrelated problem: the same permissiveness that helped IBCF's Precision@10 in the full-data regime is directly implicated in its instability under data scarcity. This is worth stating plainly rather than treating as an embarrassing flaw to minimize — it's a genuine, now-quantified cost of the Week 8 deployment choice.

### 5.6 Practical implication

`min_support=1` was selected in Week 8 for its Precision@10 advantage when the *full* training set is available. This week's analysis shows that same setting **degrades gracefully-worse, not gracefully-better, under data scarcity** — RMSE gets measurably worse, not just "less good," as user history shrinks through the 10-20 rating range, before eventually recovering once enough data accumulates. Anyone considering deploying this exact IBCF configuration (k=30, min_support=1) in a genuine cold-start-heavy production scenario (e.g. a growing catalog with many thinly-rated new users) should weigh this against the full-data Precision@10 gain it was chosen for — the RMSE-optimal min_support=3 configuration is a safer choice if new-user/sparse-history traffic is expected to be significant, at the cost of the smaller full-data Precision@10 advantage documented in Week 8.

## 6. Limitations of this simulation approach

1. **This is not real cold-start.** Every "cold" user in this simulation is actually a user who chose to rate 20+ movies (Week 3's filter), and who therefore has a genuine, coherent taste signal even when only 1-3 of their ratings are visible to the model. A real new user with 1 rating has provided that one rating essentially at random relative to their broader taste; a truncated-but-real user's 1 visible rating is a real (if incomplete) sample of a taste profile the underlying ground truth already reflects in the untouched test set. This likely makes the simulation's low-truncation-level results *more optimistic* than genuine new-user cold-start would be.
2. **Random truncation ignores rating order/recency.** A real new user's early ratings arrive in some order, and future recommender behaviour has to work with only the ratings made so far; this simulation randomly samples from a user's *entire* rating history regardless of timestamp, which could include ratings a genuinely new user wouldn't yet have made.
3. **The sparsity buckets are coverage-dependent on hyperparameters, not just the algorithm.** As shown in §3, IBCF's near-total absence from the low-neighbour buckets is a direct consequence of its tuned `min_support=1` — a stricter `min_support` (like UBCF's 10) would likely reshuffle IBCF's predictions across the buckets and could change the picture entirely. The bucket results describe *this specific tuned configuration*, not an inherent property of item-based CF in general.
4. **No cold-*item* (new movie) simulation was performed**, only cold-*user*. Truncating training rows per user changes both which users have thin histories and, incidentally, which movies lose ratings, but the truncation was not designed or targeted to isolate the new-movie case specifically.
5. **The IBCF non-monotonicity (§2, §4) was observed and discussed but not root-caused.** The proposed mechanism (noisy item pairs passing a permissive min_support filter at moderate truncation levels) is a plausible hypothesis based on the data available, not a verified explanation — a dedicated follow-up analysis of the similarity matrix's structure at each truncation level would be needed to confirm it.
