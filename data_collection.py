import os

import pandas as pd
from dotenv import load_dotenv
from gridstatusio import GridStatusClient

load_dotenv()

GRIDSTATUS_API_KEY = os.environ["GRIDSTATUS_API_KEY"]
client = GridStatusClient(api_key=GRIDSTATUS_API_KEY)

ZONE = "DOM"  # PJM Dominion zone


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
        "da_congestion": da["congestion"],
        "rt_congestion": rt["congestion"],
    })
    out["dart"] = out["da_lmp"] - out["rt_lmp"]
    return out


if __name__ == "__main__":
    df = get_dart_data(ZONE, "2025-07-01", "2025-07-07")
    print(df)
