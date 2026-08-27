# Scalability Analysis (Week 13)

Measures how UBCF and IBCF fit time and prediction latency scale with training data size, and reconciles this with the "UBCF ~5-6min/fit, IBCF ~40-70s/fit" figures reported back in Week 8.

## 1. Empirical fit-time scaling trend

Method: `scripts/analyze_scalability.py` — 4 random samples of `data/processed/train.csv` at fractions [0.25, 0.5, 0.75, 1.0] (`numpy.random.default_rng(42)`, no replacement), timing `model.fit()` wall-clock only (not evaluation) for both tuned models at each fraction.

| Model | Fraction | n_ratings | Fit time (s) |
|---|---|---|---|
| UBCF | 0.25 | 13,403 | 0.1043 |
| IBCF | 0.25 | 13,403 | 0.0350 |
| UBCF | 0.50 | 26,807 | 0.1231 |
| IBCF | 0.50 | 26,807 | 0.0353 |
| UBCF | 0.75 | 40,210 | 0.1300 |
| IBCF | 0.75 | 40,210 | 0.0269 |
| UBCF | 1.00 | 53,614 | 0.1340 |
| IBCF | 1.00 | 53,614 | 0.0252 |

**The trend is neither linear nor quadratic in `n_ratings` — fit time is nearly flat for both models.** Across a 4x increase in ratings (13,403 → 53,614), UBCF's fit time grows only ~1.28x (0.1043s → 0.1340s), and IBCF's actually stays flat-to-slightly-decreasing (0.0350s → 0.0252s, well within measurement noise at this sub-40ms scale).

The reason is structural, not a fluke: at every fraction tested, `n_users` stayed fixed at 566 and `n_movies` stayed at 1,283-1,286 (see the fit logs — even the 25% sample still covers essentially the entire user/movie catalog). That's a direct consequence of Week 3's `min_user_ratings=20` filter: with every user already guaranteed 20+ ratings, a 25% random sample of rows is overwhelmingly likely to still include at least one rating from every user. Since UBCF's similarity computation is `O(n_users²)` and IBCF's is `O(n_movies²)` — driven by the *catalog size*, not the raw rating count — and the catalog size was already saturated at the smallest fraction tested, neither model's fit time has real room to scale up in this experiment. The modest UBCF increase likely reflects the mild extra cost of each pairwise correlation having more co-rated items to average over, not a change in the number of pairs being computed.

## 2. Theoretical complexity vs empirical fit time — the apparent contradiction

Theoretically, IBCF's similarity matrix is *larger*: `O(n_movies²) = 1286² = 1,653,796` entries vs UBCF's `O(n_users²) = 566² = 320,356` entries — a ~5.16x difference. Yet empirically, IBCF's `fit()` is consistently **3-5x faster in wall-clock time** than UBCF's (0.025-0.035s vs 0.104-0.134s).

This isn't a contradiction once the implementation is accounted for, and this week's data pins the cause down precisely:

- **IBCF's similarity matrix is computed via a single vectorized numpy operation** — `cosine_similarity_matrix()` in `src/utils/similarity.py` is `normed @ normed.T`, a dense BLAS-backed matrix multiply. Despite producing 5x more entries, this is extremely fast because it's contiguous, SIMD-vectorized, single-call computation.
- **UBCF's similarity matrix is computed via `ratings_matrix.T.corr(method="pearson", min_periods=...)`** — pandas' `.corr()` with a `min_periods` threshold does not reduce to one clean BLAS call; it's a substantially less-optimized pairwise computation. A smaller theoretical matrix ends up taking longer in practice.

Critically, this week's numbers also confirm **fit time was never the source of UBCF's Week 8 slowness in the first place.** Both models' `fit()` calls complete in well under a second even on the full dataset. The multi-minute Week 8 per-config times (UBCF ~5-6min, IBCF ~40-70s) measured `fit()` *plus* `evaluate_ranking()`, and `evaluate_ranking()` calls `recommend()` for every user, which in turn calls `predict()` once per unrated candidate movie (hundreds of calls per user). §3 below shows precisely why that repeated-`predict()` cost is so lopsided between the two models.

## 3. Single-prediction latency comparison

Method: 20 repeated `predict()` calls on the same (user_id=1, movie_id=1) pair, on the full-data fitted models (`reports/scalability_prediction_latency.txt`):

| Model | Avg latency per `predict()` call |
|---|---|
| UBCF | 0.5933 ms |
| IBCF | 0.0644 ms |

**UBCF's `predict()` is ~9.21x slower per call than IBCF's.** This lines up with §2's explanation: UBCF's `predict()` does a chain of pandas Series operations (`.drop()`, `.dropna()`, boolean masking, `.abs().nlargest()`) with real per-call overhead, while IBCF's `predict()` operates on plain numpy arrays (`np.argsort`, boolean indexing) with far less overhead per call.

This also lets us sanity-check the Week 8 numbers directly: `recommend()` scores roughly 900-1,000 unrated candidate movies per user (1,286-movie catalog minus ~20-30 rated). At UBCF's 0.5933ms/call, that's roughly 900 × 0.5933ms ≈ 0.53s of `predict()` overhead per user, times 566 users ≈ **~5.0 minutes** — closely matching Week 8's observed ~5-6 minutes. At IBCF's 0.0644ms/call, the same arithmetic gives roughly 900 × 0.0644ms ≈ 0.058s per user, times 566 users ≈ **~33 seconds** — in the same range as Week 8's observed ~40-70s. The per-call latency measured this week is sufficient, on its own, to reconstruct the Week 8 timing gap.

## 4. Practical implication: scaling to a larger catalog

The two models scale in opposite directions depending on what grows:

- **More users, same movie catalog: IBCF scales better.** IBCF's similarity matrix is `O(n_movies²)` and is completely unaffected by user-base growth — adding more users only adds rows to the ratings matrix IBCF reads from, not new work for its `fit()` step. UBCF's similarity matrix is `O(n_users²)`, so a larger user base means quadratically more work to fit, and (per §3) each `predict()` call also does pandas operations whose cost grows with the size of the user-similarity structures involved.
- **More movies, same user base: UBCF scales better.** UBCF's `O(n_users²)` similarity computation is completely unaffected by catalog size. IBCF's `O(n_movies²)` similarity matrix, by contrast, grows quadratically with the number of movies — this is the well-known scaling weakness of dense item-based CF at large-catalog scale (production systems at that scale typically move to approximate/sparse neighbour methods for exactly this reason). Note this describes `fit()` cost only; IBCF's per-prediction latency advantage from §3 would still make its actual serving-time cost lower unless/until the catalog grows large enough for the `O(n_movies²)` fit cost to dominate.

In short: neither model is a universally better choice for "scale" in the abstract — the answer depends on which dimension (users or catalog) is expected to grow. For this specific dataset (566 users, 1,286 movies, growth expected primarily in users as adoption increases rather than in a fixed MovieLens-style catalog), IBCF's `O(n_movies²)`-only fit cost combined with its dramatically lower per-prediction latency (§3) makes it the more scalable choice in practice, despite UBCF's better RMSE (Week 9-10) and IBCF's own documented cold-start fragility (Week 12).
