# PJM DART Spread Forecasting

A learning project on forecasting the **DART spread** (Day-Ahead minus Real-Time LMP) for a PJM zone, built on the [GridStatus](https://www.gridstatus.io/) API.

This is a from-scratch dive into energy market data as someone new to the domain but coming from a data science background. The goal is a small, honest, reproducible pipeline — not a claim of trading alpha.

## What is DART, and why does it matter?

PJM runs two markets for the same hour of power: a **day-ahead (DA)** market that clears the afternoon before, and a **real-time (RT)** market that settles as power actually flows. The difference between them, `DART = DA price - RT price`, is what virtual traders bid on:

- An **INC** (incremental offer) is paid the DA price and buys back the equivalent volume at the RT price — it profits when `DART > 0`.
- A **DEC** (decrement bid) does the reverse — it profits when `DART < 0`.

Predicting DART is therefore directly predicting the profitability of these virtual trades. This project only builds the forecast; it does not place or simulate trades.

Prediction must happen before PJM's day-ahead market closes for the target operating day (referenced in discussion as roughly noon Eastern Prevailing Time — not independently verified). At that point no real-time price for the target day exists yet, which constrains what features are legitimate to use. Once the spread is known, its sign determines the trade direction — trading/execution logic itself is a later phase, out of scope here.

## Why the Dominion (DOM) zone

Originally framed around PJM Western Hub. After discussing why Western Hub (highest liquidity, most heavily virtual-traded, and a load-weighted aggregate across multiple pricing nodes rather than a single physical location) might actually be the hardest place to find a first signal, the project moved to the **PJM Dominion zone (DOM)**. Northern Virginia's data-center load growth has been driving real, well-documented transmission congestion inside Dominion, giving the DART spread here an actual physical story to point to, rather than being an arbitrary pick.

## Status

This project is in progress and will be updated as it grows. Current state:

- [x] Data collection: pull hourly day-ahead and real-time LMPs for the Dominion zone and compute DART (`scripts/data_collection.py`)
- [x] Leak-free feature pull (PJM's `pjm_load_forecast_hourly_historical` dataset, so features only use information that would have actually been available before the day-ahead market closed) (`scripts/data_collection.py`)
- [ ] Calendar/seasonal features
- [ ] Baselines: persistence and seasonal-naive
- [ ] Two models trained on load-forecast + calendar features: a linear model (Ridge/Lasso) and a gradient-boosted model (LightGBM)
- [ ] Walk-forward backtest (expanding time window — never a random train/test split on time series data) scoring MAE/RMSE for both models against both baselines
- [ ] Results and honest write-up of what worked and what didn't (repo README + a short blog-style post)

## A deliberate constraint: no look-ahead

The hardest part of this problem isn't the model, it's the data discipline. At the moment an INC/DEC decision would actually be made, no real-time price for that day exists yet. Every feature used here has to be something that was genuinely knowable before the day-ahead market closed — which is why load *forecasts* are used instead of load *actuals*, pulled from GridStatus's historical-vintage forecast dataset rather than the latest revision.

## Setup

```bash
python -m venv env
source env/bin/activate
pip install -r requirements.txt
```

Add your GridStatus API key to a `.env` file in the project root:

```
GRIDSTATUS_API_KEY=your_key_here
```

Then run:

```bash
python scripts/data_collection.py
```

---

## Spec

Compiled from the grilling session that shaped this project. Anything not explicitly discussed is left out rather than guessed at.

### Data Sources

All via the GridStatus API (`gridstatusio` Python client):

- `pjm_lmp_day_ahead_hourly` — PJM day-ahead LMP, filtered to `location="DOM"`
- `pjm_lmp_real_time_hourly` — PJM real-time LMP, filtered to `location="DOM"` (already hourly, no resampling needed)
- Columns discussed/used: `interval_start_utc`, `location`, `lmp`, `energy`, `congestion`, `loss`
- `pjm_load_forecast_hourly_historical` — the leak-free load forecast feature source: it's grouped by the date/time the forecast was actually made (`publish_time_utc`), so it can be pulled as it existed before DA close rather than using revised/actual values. Implemented in `scripts/data_collection.py` (`get_load_forecast_data`): filters to vintages published before each operating day's DA close and keeps the most recent one per hour. The zone value is a column name in this dataset (`dom`), not a `filter_column`/`filter_value` pair like the LMP datasets.
- The `"DOM"` filter value was confirmed working by the user after running the script locally.
- No weather forecast dataset was found in GridStatus's catalog (only load forecasts). Using weather would require an external source — not committed, listed as a v2 idea only.
- Earlier draft code used ERCOT (`ercot_spp_day_ahead_hourly`, `ercot_spp_real_time_15_min`, node `HB_NORTH`) with 15-minute RT data resampled to hourly. This was scratch/leftover code, explicitly replaced with the PJM version above — it is not part of the current plan.

### Key Decisions

- **Purpose**: an open-source personal/portfolio project — to build energy domain expertise as a beginner (strong data science background, new to energy) and to demonstrate genuine interest and GridStatus API use for an internship application. Explicitly *not* a claimed live trading system, and the internship motivation is explicitly not to be stated in the project's public materials (README etc.).
- **Feasibility stance**: the user confirmed this is currently exploratory — it is not yet known whether a real predictive edge exists at this zone. The project should behave like a feasibility test, not a project that assumes it already has a working signal.
- **ISO / zone**: PJM only for v1, Dominion (DOM) zone.
- **Leakage protection**: features must only use information that would have genuinely been knowable before DA market close. This is why the load-forecast (historical-vintage) dataset is used instead of load actuals.
- **Baselines to beat**: persistence (today's same-hour DART repeated as tomorrow's prediction) and seasonal-naive (historical average for the same hour-of-week).
- **Observation period**: training + walk-forward backtest window is `2019-01-01` to `2025-12-31`. Start date chosen (not data-availability-constrained — GridStatus dataset metadata confirms DA LMP, RT LMP, and `pjm_load_forecast_hourly_historical` all have DOM data back to `2011-01-01`) to avoid mixing in a pre-2018 PJM market era that predates the Dominion data-center congestion story, while still giving ~2 years of pre-spike baseline before the 2019-2025 transition period. Within this window, `2022` — whose DOM congestion is dominated by Winter Storm Elliott (Dec 23–25, 2022; `mean_abs_dart` for that December alone was ~93.6 $/MWh vs. ~8–50 for every other month that year) — is deliberately **included** and backtested like any other year rather than excluded, to avoid cherry-picking the dataset toward the hypothesis.
- **Final holdout**: `2026-01-01` through present is reserved untouched — not used in EDA, backtesting, or tuning until the very end of the project.
- **Warm-up**: the walk-forward backtest scores no predictions before `2020-01-01`. All of `2019` is used purely as training history, so the model has seen a full seasonal cycle (every hour-of-day/day-of-week/month combination at least once) before any prediction is judged.
- **Evaluation method**: walk-forward / expanding-window backtest. Explicitly never a random train/test split, since this is time series data.
- **Success metric**: MAE / RMSE against both baselines, evaluated inside the walk-forward backtest. A negative result (model doesn't beat baselines) is a valid, reportable outcome — not a failure condition for the project.
- **v1 feature set**: `pjm_load_forecast_hourly_historical` plus calendar/seasonal features (hour-of-day, day-of-week, month). Weather features explicitly excluded from v1.
- **v1 model**: both a linear model (e.g. Ridge/Lasso) and a gradient-boosted model (e.g. LightGBM) are fit and compared against each other and against the two baselines in the same backtest — the linear/GBM comparison is itself a reported result, not a pre-decided choice.
- **v1 scope boundary**: single PJM zone, no congestion/energy/loss decomposition of LMP, no multi-ISO comparison. These were discussed as possible directions but explicitly deferred to a stated v2 roadmap.
- **Deliverable**: a GitHub repository with a README documenting methodology, results, and honest limitations (including a negative result, if that's what the backtest shows), plus a short blog-style write-up summarizing findings.
- **Repo rename**: repo will be renamed from `dart-spread-western-hub-pjm` to match the Dominion-zone scope (proposed: `dart-spread-pjm-dom`) — rename itself is a separate action, not yet executed.

### Open Questions

- **Does an edge actually exist?** Not yet tested. This is the central open question the v1 pipeline is meant to answer.
- **Weather features**: whether/how to bring in an external weather forecast source. Named as a gap and a v2 idea, not decided either way.
- **Multi-zone / multi-ISO expansion**: raised as a possible stronger narrative (matching GridStatus's own multi-ISO value proposition) but not committed to for v1.
- **LMP decomposition**: whether to model energy/congestion/loss components separately instead of the aggregate DART spread. Raised, not resolved.
- **Exact DA market close time**: referenced in discussion as approximately noon Eastern Prevailing Time, but not independently verified.

### Glossary

- **DART**: Day-Ahead minus Real-Time price spread (`DA − RT`), computed hourly for a given PJM zone.
- **DA (Day-Ahead market/price)**: PJM's market that clears the afternoon before the operating day.
- **RT (Real-Time market/price)**: PJM's market that settles as power actually flows.
- **INC (incremental offer)**: a virtual supply offer paid the DA price, bought back at the RT price. Profits when `DART > 0`.
- **DEC (decrement bid)**: a virtual demand bid; the reverse of an INC. Profits when `DART < 0`.
- **LMP**: the price field used from PJM's day-ahead and real-time datasets to compute DART (`da_lmp`, `rt_lmp` in the code).
- **Hub**: a load-weighted aggregate price across multiple pricing nodes — not a single physical location (discussed re: Western Hub).
- **Zone**: a PJM location grouping distinct from a hub or an individual node — e.g. Dominion (`DOM`), the zone used in this project.
- **Look-ahead bias / leakage**: using data in a feature or backtest that would not actually have been available at the point a real prediction had to be made — the reason load *forecasts* (historical vintage) are used instead of load actuals.
- **Walk-forward / expanding-window backtest**: a time-series evaluation method where the training window expands forward in time, as opposed to a random train/test split.
- **Persistence baseline**: predicting tomorrow's DART as equal to today's same-hour DART.
- **Seasonal-naive baseline**: predicting DART as the historical average for that same hour-of-week.

---

## Model

Describes how the v1 model works: input → steps → output. This is a design document, not implementation — no code, no library names beyond what's already fixed above.

### Prediction Unit

One row = one **(zone, hour)** instance. A single model is trained across all historical hours for the DOM zone, with hour-of-day, day-of-week, and month as features — it is not 24 separate per-hour models, and not a single model that outputs a whole day's 24 values at once.

At inference time, predicting the next operating day means running this same per-hour model 24 times (once per target hour), not one call that returns a vector.

### Input

**Training row** (one per historical hour, only for hours where the actual DART is already known):

| Field | Source | Notes |
|---|---|---|
| `dart` (target) | `get_dart_data` (`da_lmp − rt_lmp`) | The value being predicted. Used raw — no clipping/winsorizing of the target itself. (The *historical* `dart`/`rt_lmp` used to build lag/rolling features below excludes hours past an extreme-value threshold; see "Outlier masking" below.) |
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

#### Leakage protection: why lags are cutoff-anchored, not target-hour-anchored

A trader submits INC/DEC bids for all 24 hours of the next operating day in one batch, before a single fixed DA-close cutoff (~noon ET the day before). That means the real lookback window isn't a flat N hours before each target hour — it ranges from ~12 hours (target hour 00:00) to ~35 hours (target hour 23:00), depending on which hour of the day is being predicted. A naive "N hours before this row's own timestamp" lag (e.g. `dart_lag_1` = 1 hour before target) is therefore leaky for nearly every hour of the day; a naive same-hour-yesterday lag (`dart_lag_24`) is leaky for roughly the afternoon/evening half of the day.

The fix used for the short lags (1/2/3/24h) and the rolling stats: anchor to the fixed daily cutoff instead of the row's own timestamp. Concretely, `ref_hour` = the latest hour whose DART/RT LMP is settled by DA close (assumed to be immediately at hour-end, since — unlike `load_forecast` — the LMP datasets carry no publish-time vintages to check against). `dart_lag_N` (N=1,2,3,24) = the DART value `N-1` hours before `ref_hour`. All 24 rows of a given target operating day share the *same* `ref_hour` and therefore the same lag/rolling values — only the calendar features and the hour-specific `load_forecast` vintage vary within a day.

The 48h/168h lags don't need this: a lookback of 48h or 168h clears the worst-case 35h requirement for *every* target hour of the day, so they're anchored to the literal target hour instead (`dart_lag_168` = the DART value exactly 168 hours before that row's own timestamp) — this varies per hour rather than being flat across the day, and captures the specific hour's weekly/48h pattern instead of a whole-day average. Implemented in `scripts/features.py`.

#### Outlier masking: extreme-value threshold, not specific event dates

Some hours see PJM emergency conditions (load-shed, RT price caps) that DA prices — set the day before — have no way to see coming, producing `dart` swings an order of magnitude beyond the rest of the series: -3977 and -3961 on 2022-12-23/24 (Winter Storm Elliott), but also -2785 on 2022-06-13 and -1533 on 2025-06-24 (see `analysis/eda.ipynb`, "Most extreme DART hours" and the feature-dataset section below it). Left untreated, each such hour dominates the `dart_lag_*`/`rt_lmp_lag_*`/`dart_mean_*`/`dart_std_*` features of every row whose lookback window happens to touch it (up to 168h later) — and hardcoding one event's dates doesn't generalize to the others, or to whatever the next one turns out to be.

Decision: `scripts/features.py`'s `build_features` masks `dart` and `rt_lmp` to NaN for any hour where `|dart| > 1000` (≈18x the series' overall std of ~57) before computing any lag or rolling feature from them; rows whose features end up NaN as a result are dropped by the same `dropna()` used for missing `load_forecast`/warm-up rows below. This only affects *inputs* — the raw `dart` target value for these hours is untouched, so the model is still trained and evaluated against what the spread actually did.

#### Missing data

289 hours of `load_forecast` are missing across 2019-2025 — not scattered randomly, but in ~13 near-full-day blackout windows. 8 of those align exactly with the US DST transition Sunday, one per year every year 2019-2024 (spring-forward), suggesting a systematic gap in PJM/GridStatus's forecast-vintage publishing around the clock change (not confirmed against raw pre-join data). The remaining ~5 windows don't correlate with DST and look like genuine source outages. Decision: **drop** these rows rather than impute — `scripts/features.py`'s `build_features` drops any row with a missing feature value (this also naturally drops the first ~8 days of 2019, before enough history exists for the 168-hour lag/rolling features; harmless, since all of 2019 just becomes part of the training window under the train/test split below).

### Steps

1. **Assemble** — join `dart` (training only) and `load_forecast` on `interval_start_utc` for the zone.
2. **Derive features** — calendar features (cyclical hour/day-of-week, one-hot month, is_weekend) plus cutoff-anchored `dart`/`rt_lmp` lag and rolling features, all in US/Eastern; see "Input" above. Implemented in `scripts/features.py`.
3. **Split** — single train/test split, not walk-forward: every model trains once on all of `2019-01-01`–`2025-12-31` (see "Key Decisions" above), then is scored once against the entire `2026-01-01`-onward holdout. Never a random split, and never retrained within the holdout period — a one-shot final evaluation, not a daily-retrain simulation of live trading.
4. **Fit** — two independent candidate models trained on the same feature set and target: a linear model (Ridge/Lasso) and a gradient-boosted model (LightGBM). Same feature representation for both; the one exception is that continuous features are standardized before Ridge/Lasso only (fit once on the training window alone, 2019-2025, then applied unchanged to every holdout prediction), since GBM tree splits are invariant to monotonic scaling anyway — not a difference in feature representation, just numerical preprocessing.
5. **Predict** — each fitted model outputs one DART value per (zone, hour) row across the entire 2026 holdout.
6. **Score** — compute error for both candidate models and both baselines (persistence, seasonal-naive) over the whole holdout, against the now-known actual `dart`.

### Output

Per (zone, hour): a **point estimate of DART** (continuous, $/MWh) — one number from the linear model, one from the GBM model, alongside the two baseline values, for the same hour. No trade direction (INC/DEC), no confidence interval, no PnL — those are explicitly out of scope for this model per the Model Goal above. Sign and magnitude of the raw prediction are left for a later phase to interpret.

Aggregated over the 2026 holdout, the output is a table of MAE/RMSE per model (linear, GBM, persistence, seasonal-naive) — the artifact that answers whether either candidate model beats both baselines.

### Not Yet Decided

- **RT LMP settlement lag assumption** — `scripts/features.py` assumes an hour's DART/RT LMP is knowable immediately at hour-end for the purposes of the cutoff-anchored lag features, since the LMP datasets carry no publish-time vintage to verify against (unlike `load_forecast`). Not independently confirmed against PJM's actual RT settlement/publication timeline.

## Roadmap (v2 ideas)

- Weather features from an external source (GridStatus does not currently offer a weather-forecast dataset)
- Comparing DART dynamics across multiple PJM zones, or across ISOs (ERCOT, MISO) — GridStatus's core value is unified access across all of them
- Decomposing LMP into energy/congestion/loss components rather than modeling the aggregate spread
