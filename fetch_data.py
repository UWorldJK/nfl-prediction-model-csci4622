import nflreadpy as nfl
import pandas as pd
import os

os.makedirs("data/raw", exist_ok=True)

SEASONS = list(range(2020, 2025))

# nflreadpy returns Polars DataFrames; .to_pandas() converts for the rest of the pipeline

print("Fetching weekly player stats...")
weekly = nfl.load_player_stats(SEASONS).to_pandas()
weekly.to_parquet("data/raw/weekly_stats.parquet", index=False)
print(f"  {len(weekly)} rows, {len(weekly.columns)} columns")

print("Fetching rosters...")
rosters = nfl.load_rosters_weekly(SEASONS).to_pandas()
rosters.to_parquet("data/raw/rosters.parquet", index=False)
print(f"  {len(rosters)} rows")

print("Fetching schedules...")
schedules = nfl.load_schedules(SEASONS).to_pandas()
schedules.to_parquet("data/raw/schedules.parquet", index=False)
print(f"  {len(schedules)} rows")

print("Fetching snap counts...")
snaps = nfl.load_snap_counts(SEASONS).to_pandas()
snaps.to_parquet("data/raw/snap_counts.parquet", index=False)
print(f"  {len(snaps)} rows")

print("Fetching injuries...")
injuries = nfl.load_injuries(SEASONS).to_pandas()
injuries.to_parquet("data/raw/injuries.parquet", index=False)
print(f"  {len(injuries)} rows")

print("Fetching next gen stats (passing)...")
ngs_pass = nfl.load_nextgen_stats(stat_type='passing', seasons=SEASONS).to_pandas()
ngs_pass.to_parquet("data/raw/ngs_passing.parquet", index=False)
print(f"  {len(ngs_pass)} rows")

print("Fetching next gen stats (rushing)...")
ngs_rush = nfl.load_nextgen_stats(stat_type='rushing', seasons=SEASONS).to_pandas()
ngs_rush.to_parquet("data/raw/ngs_rushing.parquet", index=False)
print(f"  {len(ngs_rush)} rows")

print("Fetching next gen stats (receiving)...")
ngs_rec = nfl.load_nextgen_stats(stat_type='receiving', seasons=SEASONS).to_pandas()
ngs_rec.to_parquet("data/raw/ngs_receiving.parquet", index=False)
print(f"  {len(ngs_rec)} rows")

print("\nDone. All data saved to data/raw/")
