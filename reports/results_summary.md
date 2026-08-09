# Movie Recommendation System — Consolidated Results Summary

Covers Weeks 1–10: data pipeline, baselines, collaborative filtering (UBCF/IBCF),
hyperparameter tuning, ranking evaluation, and statistical significance testing.

## 1. Results table

All 5 models, evaluated on the same `data/processed/test.csv` split (source:
`reports/full_model_comparison.csv`, sorted by RMSE ascending):

| Model | RMSE | MAE | Precision@10 | Recall@10 |
|---|---|---|---|---|
| UBCF (k=20, min_support=10) | 0.8430 | 0.6407 | 0.0499 | 0.0453 |
| IBCF (k=30, min_support=1) | 0.8769 | 0.6731 | 0.0562 | 0.0382 |
| UserMeanBaseline | 0.9128 | 0.7064 | 0.0250 | 0.0246 |
| ItemMeanBaseline | 0.9253 | 0.7152 | 0.0365 | 0.0346 |
| GlobalMeanBaseline | 0.9990 | 0.8000 | 0.0250 | 0.0246 |

## 2. Key findings

1. **Both tuned CF models substantially beat the weakest baseline on RMSE.** UBCF achieves RMSE 0.8430 — a 15.6% reduction versus GlobalMeanBaseline's 0.9990 ((0.9990 − 0.8430) / 0.9990). IBCF achieves RMSE 0.8769, a 12.2% reduction.

2. **UBCF and IBCF optimize different things.** UBCF has the lowest RMSE of all 5 models (0.8430), but IBCF has the highest Precision@10 of all 5 models (0.0562, vs UBCF's 0.0499).

3. **UBCF's RMSE advantage over IBCF is statistically significant, not noise.** A paired bootstrap over 1,000 resamples (`scripts/statistical_comparison.py`) found UBCF's RMSE lower than IBCF's in 100.0% of resamples, with a mean RMSE difference (IBCF − UBCF) of 0.0335 and a 95% CI of [0.0252, 0.0418] — entirely above zero. See §3 below.

4. **RMSE-optimal and Precision@10-optimal hyperparameters disagree for both models** (Week 8 grid search, `reports/hyperparameter_tuning_results.csv`):
   - UBCF: best-RMSE config is k=40/min_support=5 (RMSE 0.8371, P@10 0.0406); best-Precision@10 config (the one actually deployed) is k=20/min_support=10 (RMSE 0.8430, P@10 0.0499).
   - IBCF: best-RMSE config is k=20/min_support=3 (RMSE 0.8729, P@10 0.0522); best-Precision@10 config (the one actually deployed) is k=30/min_support=1 (RMSE 0.8769, P@10 0.0562).
   - This confirms rating-prediction accuracy and ranking quality are genuinely different objectives for this dataset, not just noisy estimates of the same thing.

5. **Among the two baselines that use partial personalization, the RMSE and ranking rankings flip.** UserMeanBaseline has better RMSE (0.9128) than ItemMeanBaseline (0.9253), but ItemMeanBaseline has better Precision@10 (0.0365 vs 0.0250) and Recall@10 (0.0346 vs 0.0246). A flat per-user average predicts individual ratings more accurately than item popularity does, but item popularity produces a more useful ranking than a user's own flat average does (which, as shown in Week 7, ties every candidate movie to the same score and degenerates to movieId-order).

6. **GlobalMeanBaseline is confirmed as the weakest model on every metric measured** — highest RMSE (0.9990) and MAE (0.8000), tied-lowest Precision@10/Recall@10 (0.0250/0.0246) with UserMeanBaseline. This validates that both CF models are learning genuine collaborative signal rather than reproducing a summary statistic.

## 3. Statistical significance (Week 10)

Method: paired bootstrap over per-prediction squared errors on the test set (`get_squared_errors` + `bootstrap_rmse_comparison` in `scripts/statistical_comparison.py`), 1,000 resamples, `numpy.random.default_rng(42)`. "Paired" means the same resampled row indices are applied to both models' error arrays on each iteration, so the comparison controls for the fact both models were scored on identical test rows.

- Point-estimate RMSE: UBCF 0.8430, IBCF 0.8769
- UBCF had lower RMSE in **100.0%** of 1,000 resamples
- Mean RMSE difference (IBCF − UBCF): **0.0335**
- 95% CI: **[0.0252, 0.0418]**

**Conclusion:** the 95% CI for (IBCF RMSE − UBCF RMSE) is entirely positive and excludes zero, so UBCF's RMSE advantage over IBCF at these tuned hyperparameters is **statistically significant**, not attributable to resampling variance.

## 4. Relevant history from earlier weeks

- **Week 6 — IBCF mean-centering bug.** `ItemBasedCF.predict()` originally aggregated *raw* neighbour ratings weighted by *signed* similarities, so items with strongly negative-similarity neighbours got dragged toward the 0.5 floor — 45% of a 20-row diagnostic sample clipped to exactly 0.5, and full-model RMSE was 2.5031. Fixed by switching to item-mean-centred deviations (`prediction = target_item_mean + Σ(sim·(rating − item_mean)) / Σ|sim|`), which brought RMSE down to ~0.87–0.90 depending on hyperparameters — in the range now confirmed by both Week 8 tuning and this report's results table.
- **Week 7 — UBCF prediction saturation.** UBCF's mean-centred (Resnick-style) formula is mathematically unbounded, so ~1.07% of raw predictions legitimately exceeded 5.0 before clipping in the pre-tuning configuration, concentrated in high-mean users (one sampled user saturated at a 4.1% rate vs the 1.07% baseline). This produced large tied-at-5.0 blocks in `recommend()` output, which in turn caused non-monotonic Precision@K for UBCF (Precision@10 briefly exceeding Precision@5). A deterministic tie-breaker (secondary sort key: neighbour count; tertiary: movieId) was added for reproducibility, but did **not** resolve the non-monotonicity itself — see Limitations below.

## 5. Open limitations to address in Weeks 11–12

1. **Cold-start behaviour has not been analyzed.** All evaluation so far uses the existing per-user train/test split, where every test user already has training history. No experiment yet measures how UBCF, IBCF, or the baselines perform for a user/movie with zero or near-zero prior ratings — this is a real gap for a "recommendation system" report, since GlobalMeanBaseline and the personalized-fallback baselines are effectively the only current cold-start behavior, and it's untested directly.
2. **Scalability gap between UBCF and IBCF fit time needs a formal write-up (targeted for Week 13).** From the Week 8 grid search timings, UBCF fits took roughly 4.7–6.2 minutes per configuration (282–370s, excluding one 1499s outlier likely caused by system contention) versus IBCF's roughly 0.6–1.2 minutes (38–69s) — a ~5–6x gap driven by UBCF's per-candidate pandas-based `recommend()` path versus IBCF's vectorized numpy similarity matrix. This has been observed repeatedly but never formally profiled or written up as its own analysis.
3. **UBCF's non-monotonic Precision@K was accepted as statistical noise, not resolved.** After the Week 10 tuning config (k=20, min_support=10), UBCF's Precision@K did not reliably decrease monotonically across K=5/10/20 in earlier investigation; the tie-breaker fix (Week 7) made rankings reproducible but did not change the underlying pattern, and the team's working conclusion was that absolute precision values this low (~0.04) are within normal sampling noise. This was a judgment call, not a proof, and should be revisited with a larger K sweep or significance test on Precision@K itself if it matters for the final report.
4. **Ranking metrics are operating close to the noise floor.** Precision@10 values across all 5 models range only from 0.025 to 0.056, and per-user hit-rate sampling in earlier debugging showed most individual users have zero hits in their top-20 recommendations. This is consistent with a small, sparse catalog (1,286 movies, 92.63% matrix sparsity) but means ranking-metric comparisons between models should be read as directional, not precise, without the kind of bootstrap CI applied to RMSE in this report.
5. **The Week 10 bootstrap only tested RMSE significance, not ranking-metric significance.** Whether IBCF's higher Precision@10 (0.0562 vs UBCF's 0.0499) is itself statistically significant given point 4 above has not been tested — only the RMSE comparison was bootstrapped.
