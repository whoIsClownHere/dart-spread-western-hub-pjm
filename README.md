# PJM DART Spread Forecasting

A learning project on forecasting the **DART spread** (Day-Ahead minus Real-Time LMP) for a PJM zone, built on the [GridStatus](https://www.gridstatus.io/) API.

This is a from-scratch dive into energy market data as someone new to the domain but coming from a data science background. The goal is a small, honest, reproducible pipeline — not a claim of trading alpha.

## What is DART, and why does it matter?

PJM runs two markets for the same hour of power: a **day-ahead (DA)** market that clears the afternoon before, and a **real-time (RT)** market that settles as power actually flows. The difference between them, `DART = DA price - RT price`, is what virtual traders bid on:

- An **INC** (incremental offer) is paid the DA price and buys back the equivalent volume at the RT price — it profits when `DART > 0`.
- A **DEC** (decrement bid) does the reverse — it profits when `DART < 0`.

Predicting DART is therefore directly predicting the profitability of these virtual trades. This project only builds the forecast; it does not place or simulate trades.

## Why the Dominion (DOM) zone

Northern Virginia's data-center load growth has been driving real, well-documented transmission congestion inside PJM's Dominion zone. That gives the DART spread here an actual physical story to point to, rather than being an arbitrary pick.

## Status

This project is in progress and will be updated as it grows. Current state:

- [x] Data collection: pull hourly day-ahead and real-time LMPs for the Dominion zone and compute DART (`data_collection.py`)
- [ ] Leak-free feature pull (PJM's `pjm_load_forecast_hourly_historical` dataset, so features only use information that would have actually been available before the day-ahead market closed)
- [ ] Calendar/seasonal features
- [ ] Baselines: persistence and seasonal-naive
- [ ] A simple model (gradient boosting or linear) trained on load-forecast + calendar features
- [ ] Walk-forward backtest (expanding time window — never a random train/test split on time series data) comparing the model against both baselines
- [ ] Results and honest write-up of what worked and what didn't

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
python data_collection.py
```

## Roadmap (v2 ideas)

- Weather features from an external source (GridStatus does not currently offer a weather-forecast dataset)
- Comparing DART dynamics across multiple PJM zones, or across ISOs (ERCOT, MISO) — GridStatus's core value is unified access across all of them
- Decomposing LMP into energy/congestion/loss components rather than modeling the aggregate spread
