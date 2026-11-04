import nfl_data_py as nfl
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

SEASONS = list(range(2020, 2025))

print("Fetching weekly player stats...")
weekly = nfl.import_weekly_data(SEASONS)
weekly.to_parquet("data/raw/weekly_stats.parquet", index=False)
print(f"  {len(weekly)} rows, {len(weekly.columns)} columns")

print("Fetching rosters...")
rosters = nfl.import_weekly_rosters(SEASONS)
rosters.to_parquet("data/raw/rosters.parquet", index=False)
print(f"  {len(rosters)} rows")

print("Fetching schedules...")
schedules = nfl.import_schedules(SEASONS)
schedules.to_parquet("data/raw/schedules.parquet", index=False)
print(f"  {len(schedules)} rows")

print("Fetching snap counts...")
snaps = nfl.import_snap_counts(SEASONS)
snaps.to_parquet("data/raw/snap_counts.parquet", index=False)
print(f"  {len(snaps)} rows")

print("Fetching injuries...")
injuries = nfl.import_injuries(SEASONS)
injuries.to_parquet("data/raw/injuries.parquet", index=False)
print(f"  {len(injuries)} rows")

print("Fetching next gen stats (passing)...")
ngs_pass = nfl.import_ngs_data(stat_type='passing', years=SEASONS)
ngs_pass.to_parquet("data/raw/ngs_passing.parquet", index=False)
print(f"  {len(ngs_pass)} rows")

print("Fetching next gen stats (rushing)...")
ngs_rush = nfl.import_ngs_data(stat_type='rushing', years=SEASONS)
ngs_rush.to_parquet("data/raw/ngs_rushing.parquet", index=False)
print(f"  {len(ngs_rush)} rows")

print("Fetching next gen stats (receiving)...")
ngs_rec  = nfl.import_ngs_data(stat_type='receiving', years=SEASONS)
ngs_rec.to_parquet("data/raw/ngs_receiving.parquet", index=False)
print(f"  {len(ngs_rec)} rows")

print("\nDone. All data saved to data/raw/")