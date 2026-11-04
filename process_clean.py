# process_clean.py
# Cleans raw data, engineers features, and builds position-specific datasets

import pandas as pd
import numpy as np
import os

os.makedirs("data/processed", exist_ok=True)

# ── Load raw data ──────────────────────────────────────────────
weekly   = pd.read_parquet("data/raw/weekly_stats.parquet")
rosters  = pd.read_parquet("data/raw/rosters.parquet")
schedules = pd.read_parquet("data/raw/schedules.parquet")
snaps    = pd.read_parquet("data/raw/snap_counts.parquet")

# ── Build opponent defensive strength ─────────────────────────
# Points allowed per game as a proxy for defensive strength
opp_strength = (
    schedules
    .melt(id_vars=['season', 'week', 'home_team', 'away_team',
                   'home_score', 'away_score'],
          value_vars=['home_team', 'away_team'])
    .assign(
        team=lambda df: df['value'],
        pts_allowed=lambda df: np.where(
            df['variable'] == 'home_team',
            df['away_score'], df['home_score']
        )
    )
    [['season', 'week', 'team', 'pts_allowed']]
    .dropna()
)

# Rolling 4-week average points allowed by each team (as defense)
opp_strength = opp_strength.sort_values(['season', 'team', 'week'])
opp_strength['def_pts_allowed_avg'] = (
    opp_strength
    .groupby(['season', 'team'])['pts_allowed']
    .transform(lambda x: x.shift(1).rolling(4, min_periods=1).mean())
)
opp_strength = opp_strength.rename(columns={'team': 'recent_team'})

# ── Merge snap counts ──────────────────────────────────────────
snaps_clean = snaps[['season', 'week', 'player', 'offense_pct']].copy()
snaps_clean.columns = ['season', 'week', 'player_display_name', 'snap_pct']
# ── Merge rosters (age, experience, position) ─────────────────
rosters = rosters.rename(columns={'gsis_id': 'player_id'})
rosters['age'] = rosters['season'] - pd.to_datetime(rosters['birth_date']).dt.year
roster_clean = rosters.drop_duplicates(subset=['season', 'player_id'])

# ── Merge everything onto weekly stats ────────────────────────
# nflreadpy names the team column 'team'; the merges below use 'recent_team'
df = weekly.copy().rename(columns={'team': 'recent_team'})
df = df.merge(roster_clean[['season', 'player_id', 'age', 'years_exp']],
              on=['season', 'player_id'], how='left')
df = df.merge(snaps_clean, on=['season', 'week', 'player_display_name'], how='left')
df = df.merge(opp_strength[['season', 'week', 'recent_team', 'def_pts_allowed_avg']],
              on=['season', 'week', 'recent_team'], how='left')

# ── Target variable: PPR fantasy points ───────────────────────
# Use fantasy_points_ppr if available, otherwise compute it
if 'fantasy_points_ppr' in df.columns:
    df['target'] = df['fantasy_points_ppr']
else:
    df['target'] = (
        df.get('passing_yards', 0) * 0.04 +
        df.get('passing_tds', 0) * 4 +
        df.get('interceptions', 0) * -1 +
        df.get('rushing_yards', 0) * 0.1 +
        df.get('rushing_tds', 0) * 6 +
        df.get('receptions', 0) * 1 +       # PPR: 1pt per reception
        df.get('receiving_yards', 0) * 0.1 +
        df.get('receiving_tds', 0) * 6
    )

# ── Rolling features (per player) ─────────────────────────────
df = df.sort_values(['player_id', 'season', 'week'])

roll_cols = ['target', 'snap_pct']
for col in roll_cols:
    if col in df.columns:
        # Shift 1 so we don't leak current game into features
        df[f'{col}_avg3'] = (
            df.groupby('player_id')[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
        )
        df[f'{col}_avg5'] = (
            df.groupby('player_id')[col]
            .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
        )
        df[f'{col}_std3'] = (
            df.groupby('player_id')[col]
            .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std())
        )

# ── Additional engineered features ────────────────────────────
# Target share (receiving targets / team targets)
if 'targets' in df.columns:
    team_targets = df.groupby(['season', 'week', 'recent_team'])['targets'].transform('sum')
    df['target_share'] = df['targets'] / team_targets.replace(0, np.nan)

# Air yards share
if 'air_yards' in df.columns and 'targets' in df.columns:
    team_air = df.groupby(['season', 'week', 'recent_team'])['air_yards'].transform('sum')
    df['air_yards_share'] = df['air_yards'] / team_air.replace(0, np.nan)

# Home game flag
home_games = schedules[['season', 'week', 'home_team']].copy()
home_games['is_home'] = 1
df = df.merge(
    home_games.rename(columns={'home_team': 'recent_team'}),
    on=['season', 'week', 'recent_team'], how='left'
)
df['is_home'] = df['is_home'].fillna(0)

# ── Drop rows missing target ───────────────────────────────────
df = df.dropna(subset=['target'])

# ── Filter to skill positions only ────────────────────────────
skill_positions = ['QB', 'RB', 'WR', 'TE']
df = df[df['position'].isin(skill_positions)]

# ── Drop rows where player had 0 snaps (DNP) ──────────────────
if 'snap_pct' in df.columns:
    df = df[df['snap_pct'].notna()]

# ── Save position-specific datasets ───────────────────────────
for pos in skill_positions:
    pos_df = df[df['position'] == pos].copy()
    pos_df.to_parquet(f"data/processed/{pos.lower()}_data.parquet", index=False)
    print(f"{pos}: {len(pos_df)} player-games, {len(pos_df.columns)} columns")

# Save full dataset too
df.to_parquet("data/processed/all_positions.parquet", index=False)
print(f"\nAll positions: {len(df)} rows saved to data/processed/")