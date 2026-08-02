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
| `hour_of_day` | derived from `interval_start_utc` | One-hot categorical, 24 levels. |
| `day_of_week` | derived from `interval_start_utc` | One-hot categorical, 7 levels. |
| `month` | derived from `interval_start_utc` | One-hot categorical, 12 levels. |

**Inference row** (one per hour of the target operating day, built before DA close):

Same feature columns as training (`load_forecast`, `hour_of_day`, `day_of_week`, `month`) — no `dart`, since that's what's being predicted. `load_forecast` is the vintage available as of the actual moment of prediction, not a future-revised value — this is what makes the feature leak-free at inference time by construction, not just in the backtest.

## Steps

1. **Assemble** — join `dart` (training only) and `load_forecast` on `interval_start_utc` for the zone.
2. **Derive calendar features** — extract hour-of-day, day-of-week, month from `interval_start_utc`; one-hot encode all three.
3. **Split (walk-forward)** — expanding training window, retrained daily: each simulated "day" trains on all history strictly before that day's DA close, then predicts that day's 24 hours. Never a random split. Observation period is `2019-01-01`–`2025-12-31` (see [SPEC.md](SPEC.md#key-decisions)); the first scored prediction is `2020-01-01`, with all of `2019` used as warm-up training history only. `2026-01-01` onward is a final holdout, untouched until the end of the project.
4. **Fit** — two independent candidate models trained on the same feature set and target: a linear model (Ridge/Lasso) and a gradient-boosted model (LightGBM). Same one-hot feature representation for both, for a fair comparison.
5. **Predict** — each fitted model outputs one DART value per (zone, hour) row in the held-out day.
6. **Score** — for that backtest day, compute error for both candidate models and both baselines (persistence, seasonal-naive) against the now-known actual `dart`.
7. **Advance** — roll the window forward one day and repeat from step 3 until the backtest period is exhausted.

## Output

Per (zone, hour): a **point estimate of DART** (continuous, $/MWh) — one number from the linear model, one from the GBM model, alongside the two baseline values, for the same hour. No trade direction (INC/DEC), no confidence interval, no PnL — those are explicitly out of scope for this model per [SPEC.md](SPEC.md)'s Model Goal section. Sign and magnitude of the raw prediction are left for a later phase to interpret.

Aggregated over a backtest period, the output is a table of MAE/RMSE per model (linear, GBM, persistence, seasonal-naive) — the artifact that answers whether either candidate model beats both baselines.

## Not Yet Decided

- **Feature scaling for the linear model** — one-hot columns don't need it, but whether `load_forecast` is standardized before fitting Ridge/Lasso isn't decided.
- **Missing/incomplete hours** — what happens to a row if `load_forecast` has no vintage published before that day's DA close (e.g. very start of history), or DA/RT LMP is missing for an hour.
