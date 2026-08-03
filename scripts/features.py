import numpy as np
import pandas as pd

ET = "US/Eastern"
DA_CLOSE_HOUR_ET = 12  # matches scripts/data_collection.py's da_close_hour_et

# Anchored to the fixed daily cutoff (constant across a target day's 24 rows) —
# required for anything under ~35h back, since that's the worst-case lookback
# (target hour 23:00 needs a 35h lookback; see "Leakage protection" in MODEL.md).
CUTOFF_ANCHORED_LAG_HOURS = (1, 2, 3, 24)
RT_LMP_CUTOFF_ANCHORED_LAG_HOURS = (1, 24)  # dart gets 2/3h lags too; rt_lmp doesn't

# Anchored to the literal target hour (varies per hour, e.g. "this same hour,
# exactly 7 days ago") — safe here specifically because 48h/168h both clear
# the 35h worst-case lookback for every hour of the day, unlike the shorter lags.
TARGET_ANCHORED_LAG_HOURS = (48, 168)

ROLLING_WINDOWS = (24, 168)


def _last_known_hour_utc(operating_day_et: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Latest hour whose DART/RT LMP is settled and knowable by DA close.

    Uses `DateOffset` (calendar-day arithmetic) rather than `Timedelta` (fixed
    24h duration) so the cutoff lands on the correct wall-clock hour across
    DST transitions. Assumes RT LMP is available immediately at hour-end, since
    the source dataset carries no publish-time vintages to check against
    (unlike `load_forecast`, whose leak-safety is enforced upstream in
    `data_collection.py` via actual publish timestamps).
    """
    cutoff_et = operating_day_et - pd.DateOffset(days=1) + pd.Timedelta(hours=DA_CLOSE_HOUR_ET)
    cutoff_utc = cutoff_et.tz_convert("UTC")
    return cutoff_utc - pd.Timedelta(hours=1)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the leakage-safe feature matrix from `dart_data.csv`'s joined columns.

    `df` must be indexed by `interval_start_utc` (tz-aware, complete hourly,
    as produced by `data_collection.save_data`), with `dart`, `rt_lmp`, and
    `load_forecast` columns. Rows with any missing feature (unfilled
    `load_forecast` vintage, or not enough history yet for the longest lag/
    rolling window) are dropped rather than imputed.
    """
    df = df.sort_index()
    et = df.index.tz_convert(ET)
    operating_day = et.normalize()

    out = pd.DataFrame(index=df.index)
    out["load_forecast"] = df["load_forecast"]

    hour = et.hour.to_numpy()
    out["hour_sin"] = np.sin(2 * np.pi * hour / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hour / 24)

    dow = et.dayofweek.to_numpy()  # Monday=0 ... Sunday=6
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    out["is_weekend"] = (dow >= 5).astype(int)

    month = pd.Categorical(et.month, categories=range(1, 13))
    month_dummies = pd.get_dummies(month, prefix="month")
    month_dummies.index = df.index
    out = pd.concat([out, month_dummies], axis=1)

    ref_hour = _last_known_hour_utc(operating_day)  # per-row, but constant within an operating day
    dart = df["dart"]
    rt_lmp = df["rt_lmp"]

    for n in CUTOFF_ANCHORED_LAG_HOURS:
        lookup = ref_hour - pd.Timedelta(hours=n - 1)
        out[f"dart_lag_{n}"] = dart.reindex(lookup).to_numpy()

    for n in RT_LMP_CUTOFF_ANCHORED_LAG_HOURS:
        lookup = ref_hour - pd.Timedelta(hours=n - 1)
        out[f"rt_lmp_lag_{n}"] = rt_lmp.reindex(lookup).to_numpy()

    for n in TARGET_ANCHORED_LAG_HOURS:
        out[f"dart_lag_{n}"] = dart.shift(n).to_numpy()
        out[f"rt_lmp_lag_{n}"] = rt_lmp.shift(n).to_numpy()

    rolling_mean = {w: dart.rolling(w, min_periods=w).mean() for w in ROLLING_WINDOWS}
    rolling_std = {w: dart.rolling(w, min_periods=w).std() for w in ROLLING_WINDOWS}

    out["dart_mean_24"] = rolling_mean[24].reindex(ref_hour).to_numpy()
    out["dart_std_24"] = rolling_std[24].reindex(ref_hour).to_numpy()
    out["dart_mean_168"] = rolling_mean[168].reindex(ref_hour).to_numpy()

    return out.dropna()
