from gridstatusio import GridStatusClient

GRIDSTATUS_API_KEY = os.environ["GRIDSTATUS_API_KEY"]

client = GridStatusClient(api_key=GRIDSTATUS_API_KEY)

da = client.get_dataset(
    "pjm_lmp_day_ahead_hourly",
    start="2023-01-01", end="2026-01-01",
    filter_column="location", filter_value="WESTERN HUB",
    columns=["interval_start_utc","location","lmp","energy","congestion","loss"],
)

rt = client.get_dataset(
    "pjm_lmp_real_time_hourly",
    start="2023-01-01", end="2026-01-01",
    filter_column="location", filter_value="WESTERN HUB",
    columns=["interval_start_utc","location","lmp","energy","congestion","loss"],
)

print("Day Ahead LMP Data:")
print(da.to_csv(index=False))
print('--------------------------------')
print('--------------------------------')
print('--------------------------------')
print("\nReal Time LMP Data:")
print(rt.to_csv(index=False))
