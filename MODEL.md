# MODEL — DART Spread Prediction Logic

Describes how the v1 model works: input → steps → output. This is a design document, not implementation — no code, no library names beyond what's already fixed in [SPEC.md](SPEC.md).

## Prediction Unit

One row = one **(zone, hour)** instance. A single model is trained across all historical hours for the DOM zone, with hour-of-day, day-of-week, and month as features — it is not 24 separate per-hour models, and not a single model that outputs a whole day's 24 values at once.

At inference time, predicting the next operating day means running this same per-hour model 24 times (once per target hour), not one call that returns a vector.

## Input

**Training row** (one per historical hour, only for hours where the actual DART is already known):

| Field | Source | Notes |
|---|---|---|
| `dart` (target) | `get_dart_data` (`da_lmp − rt_lmp`) | The value being predicted. Used raw — no clipping/winsorizing of outliers. |
| `load_forecast` | `get_load_forecast_data` | Leak-free: latest forecast vintage published before that hour's operating day's DA close. |
| `hour_sin`, `hour_cos` | derived from `interval_start_utc`, **US/Eastern** | Cyclical encoding of hour-of-day (24h period), not one-hot — supersedes the earlier one-hot plan. |
| `dow_sin`, `dow_cos` | derived from `interval_start_utc`, US/Eastern | Cyclical encoding of day-of-week (7-day period). |
| `is_weekend` | derived from day-of-week, US/Eastern | 0/1 flag, kept alongside the cyclical dow encoding rather than relying on it alone. |
| `month_1`...`month_12` | derived from `interval_start_utc`, US/Eastern | One-hot, 12 fixed categories (unlike hour/dow, kept one-hot). |
| `dart_lag_1/2/3/24`, `rt_lmp_lag_1/24` | trailing `dart`/`rt_lmp` history | **Cutoff-anchored** — see "Leakage protection" below. |
| `dart_lag_48/168`, `rt_lmp_lag_48/168` | trailing `dart`/`rt_lmp` history | **Target-hour-anchored** ("this exact hour, N days ago") — safe unlike the shorter lags because 48h/168h both clear the 35h worst-case lookback for every target hour, so anchoring to the literal target hour (rather than the cutoff) carries more signal without leaking. |
| `dart_mean_24`, `dart_std_24`, `dart_mean_168` | rolling `dart` history | Same cutoff-anchoring as the lag features. |

All calendar fields (`hour_sin/cos`, `dow_sin/cos`, `is_weekend`, `month_*`) are derived from `interval_start_utc` converted to **US/Eastern**, not raw UTC — this is PJM's own operating clock (DA close, operating-day boundaries), and deriving hour-of-day from UTC would misalign it with what actually drives demand/congestion patterns.

Explicitly **excluded**: `da_lmp`, `da_energy`, `da_congestion`, `da_loss` (and anything derived from them, e.g. a `da_congestion_share`) for the target hour. `dart = da_lmp − rt_lmp`, so the target day's own DA decomposition is half the label and doesn't exist yet at prediction time — including it would be direct target leakage, not a subtle timing issue.

**Inference row** (one per hour of the target operating day, built before DA close):

Same feature columns as training — no `dart`, since that's what's being predicted (the `dart_lag_*`/`dart_mean_*`/`dart_std_*` features use only already-realized history, so they're populated at inference time too). `load_forecast` is the vintage available as of the actual moment of prediction, not a future-revised value — this is what makes that feature leak-free at inference time by construction, not just in the backtest.

### Leakage protection: why lags are cutoff-anchored, not target-hour-anchored

A trader submits INC/DEC bids for all 24 hours of the next operating day in one batch, before a single fixed DA-close cutoff (~noon ET the day before). That means the real lookback window isn't a flat N hours before each target hour — it ranges from ~12 hours (target hour 00:00) to ~35 hours (target hour 23:00), depending on which hour of the day is being predicted. A naive "N hours before this row's own timestamp" lag (e.g. `dart_lag_1` = 1 hour before target) is therefore leaky for nearly every hour of the day; a naive same-hour-yesterday lag (`dart_lag_24`) is leaky for roughly the afternoon/evening half of the day.

The fix used for the short lags (1/2/3/24h) and the rolling stats: anchor to the fixed daily cutoff instead of the row's own timestamp. Concretely, `ref_hour` = the latest hour whose DART/RT LMP is settled by DA close (assumed to be immediately at hour-end, since — unlike `load_forecast` — the LMP datasets carry no publish-time vintages to check against). `dart_lag_N` (N=1,2,3,24) = the DART value `N-1` hours before `ref_hour`. All 24 rows of a given target operating day share the *same* `ref_hour` and therefore the same lag/rolling values — only the calendar features and the hour-specific `load_forecast` vintage vary within a day.

The 48h/168h lags don't need this: a lookback of 48h or 168h clears the worst-case 35h requirement for *every* target hour of the day, so they're anchored to the literal target hour instead (`dart_lag_168` = the DART value exactly 168 hours before that row's own timestamp) — this varies per hour rather than being flat across the day, and captures the specific hour's weekly/48h pattern instead of a whole-day average. Implemented in `scripts/features.py`.

### Missing data

289 hours of `load_forecast` are missing across 2019-2025 — not scattered randomly, but in ~13 near-full-day blackout windows. 8 of those align exactly with the US DST transition Sunday, one per year every year 2019-2024 (spring-forward), suggesting a systematic gap in PJM/GridStatus's forecast-vintage publishing around the clock change (not confirmed against raw pre-join data). The remaining ~5 windows don't correlate with DST and look like genuine source outages. Decision: **drop** these rows rather than impute — `scripts/features.py`'s `build_features` drops any row with a missing feature value (this also naturally drops the first ~8 days of 2019, before enough history exists for the 168-hour lag/rolling features; harmless, since all of 2019 is warm-up-only per the walk-forward split below).

## Steps

1. **Assemble** — join `dart` (training only) and `load_forecast` on `interval_start_utc` for the zone.
2. **Derive features** — calendar features (cyclical hour/day-of-week, one-hot month, is_weekend) plus cutoff-anchored `dart`/`rt_lmp` lag and rolling features, all in US/Eastern; see "Input" above. Implemented in `scripts/features.py`.
3. **Split (walk-forward)** — expanding training window, retrained daily: each simulated "day" trains on all history strictly before that day's DA close, then predicts that day's 24 hours. Never a random split. Observation period is `2019-01-01`–`2025-12-31` (see [SPEC.md](SPEC.md#key-decisions)); the first scored prediction is `2020-01-01`, with all of `2019` used as warm-up training history only. `2026-01-01` onward is a final holdout, untouched until the end of the project.
4. **Fit** — two independent candidate models trained on the same feature set and target: a linear model (Ridge/Lasso) and a gradient-boosted model (LightGBM). Same feature representation for both; the one exception is that continuous features are standardized before Ridge/Lasso only (fit on that fold's training data alone), since GBM tree splits are invariant to monotonic scaling anyway — not a difference in feature representation, just numerical preprocessing.
5. **Predict** — each fitted model outputs one DART value per (zone, hour) row in the held-out day.
6. **Score** — for that backtest day, compute error for both candidate models and both baselines (persistence, seasonal-naive) against the now-known actual `dart`.
7. **Advance** — roll the window forward one day and repeat from step 3 until the backtest period is exhausted.

## Output

Per (zone, hour): a **point estimate of DART** (continuous, $/MWh) — one number from the linear model, one from the GBM model, alongside the two baseline values, for the same hour. No trade direction (INC/DEC), no confidence interval, no PnL — those are explicitly out of scope for this model per [SPEC.md](SPEC.md)'s Model Goal section. Sign and magnitude of the raw prediction are left for a later phase to interpret.

Aggregated over a backtest period, the output is a table of MAE/RMSE per model (linear, GBM, persistence, seasonal-naive) — the artifact that answers whether either candidate model beats both baselines.

## Not Yet Decided

- **Where fold-local scaling actually lives** — decided *that* it happens (see step 4), not yet implemented: `models.py`/`backtest.py` need to fit the scaler per walk-forward fold on that fold's training data only, not on the full history.
- **RT LMP settlement lag assumption** — `scripts/features.py` assumes an hour's DART/RT LMP is knowable immediately at hour-end for the purposes of the cutoff-anchored lag features, since the LMP datasets carry no publish-time vintage to verify against (unlike `load_forecast`). Not independently confirmed against PJM's actual RT settlement/publication timeline.
