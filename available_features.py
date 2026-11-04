import inspect
import nfl_data_py as nfl
import pandas as pd
weekly = pd.read_parquet("data/raw/weekly_stats.parquet")
print([c for c in weekly.columns if 'player' in c.lower() or 'name' in c.lower()])