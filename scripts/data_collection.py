import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from gridstatusio import GridStatusClient

load_dotenv()

GRIDSTATUS_API_KEY = os.environ["GRIDSTATUS_API_KEY"]
client = GridStatusClient(api_key=GRIDSTATUS_API_KEY)

ROOT_DIR = Path(__file__).resolve().parent.parent
INPUT_DATA_DIR = ROOT_DIR / "input_data"

ZONE = "DOM"  # PJM Dominion zone

TRAIN_START = "2019-01-01"
TRAIN_END = "2026-01-01"  # exclusive: covers 2019-01-01 through 2025-12-31
WARMUP_END = "2020-01-01"  # exclusive: all of 2019 is warm-up-only training history
HOLDOUT_START = "2026-01-01"  # final holdout, untouched until the end of the project


def get_dart_data(zone: str, start: str, end: str) -> pd.DataFrame:
    """DART = day-ahead LMP minus real-time LMP, hourly, for one PJM zone.

    Positive DART favors INC offers (sell at DA, buy back cheaper at RT);
    negative DART favors DEC bids.
    """
    da = client.get_dataset(
        dataset="pjm_lmp_day_ahead_hourly",
        start=start, end=end,
        filter_column="location", filter_value=zone,
        columns=["interval_start_utc", "location", "lmp", "energy", "congestion", "loss"],
    )
    rt = client.get_dataset(
        dataset="pjm_lmp_real_time_hourly",
        start=start, end=end,
        filter_column="location", filter_value=zone,
        columns=["interval_start_utc", "location", "lmp", "energy", "congestion", "loss"],
    )

    da = da.set_index("interval_start_utc")
    rt = rt.set_index("interval_start_utc")

    out = pd.DataFrame({
        "da_lmp": da["lmp"],
        "rt_lmp": rt["lmp"],
        "da_energy": da["energy"],
        "rt_energy": rt["energy"],
        "da_congestion": da["congestion"],
        "rt_congestion": rt["congestion"],
        "da_loss": da["loss"],
        "rt_loss": rt["loss"],
    })
    out["dart"] = out["da_lmp"] - out["rt_lmp"]
    return out


def get_load_forecast_data(
    zone: str, start: str, end: str, da_close_hour_et: int = 12
) -> pd.Series:
    """Leak-free hourly load forecast for one PJM zone.

    `pjm_load_forecast_hourly_historical` carries multiple forecast vintages
    per hour (`publish_time_utc`), since PJM reissues the forecast every six
    hours. For each target hour we keep only vintages published before that
    operating day's DA market close (~noon ET, per SPEC.md — not verified)
    and take the most recent of those, so the feature reflects only what was
    actually knowable before DA close.
    """
    raw = client.get_dataset(
        dataset="pjm_load_forecast_hourly_historical",
        start=start, end=end,
        columns=["interval_start_utc", "publish_time_utc", zone.lower()],
    )
    raw = raw.rename(columns={zone.lower(): "load_forecast"})

    interval_et = raw["interval_start_utc"].dt.tz_convert("US/Eastern")
    operating_day = interval_et.dt.normalize()
    da_close = operating_day - pd.Timedelta(days=1) + pd.Timedelta(hours=da_close_hour_et)

    raw = raw[raw["publish_time_utc"] <= da_close]
    raw = raw.sort_values("publish_time_utc")
    latest = raw.groupby("interval_start_utc").last()

    return latest["load_forecast"]


def save_data(zone: str, start: str, end: str, path: str) -> None:
    """Fetch DART and load-forecast data for one zone and save the joined result to CSV."""
    dart = get_dart_data(zone, start, end)
    load_forecast = get_load_forecast_data(zone, start, end)
    dart.join(load_forecast).to_csv(path)


if __name__ == "__main__":
    save_data(ZONE, TRAIN_START, TRAIN_END, INPUT_DATA_DIR / "dart_data.csv")
