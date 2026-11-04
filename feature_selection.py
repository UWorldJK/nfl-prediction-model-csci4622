# feature_selection.py
# Computes mutual information (information gain) for each feature
# and ranks them by predictive power for the target variable

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.impute import SimpleImputer

POSITION = 'WR'   # change to QB, RB, TE as needed

# ── Load processed data ────────────────────────────────────────
df = pd.read_parquet(f"data/processed/{POSITION.lower()}_data.parquet")

# ── Define candidate features ─────────────────────────────────
CANDIDATE_FEATURES = [
    # Rolling performance
    'target_avg3', 'target_avg5', 'target_std3',
    'snap_pct_avg3', 'snap_pct_avg5',
    # Player attributes
    'age', 'years_exp',
    # Game context
    'is_home', 'week', 'def_pts_allowed_avg',
    # Receiving-specific (WR/TE)
    'target_share', 'air_yards_share',
    # Raw stats from previous game (shifted)
    'targets', 'receptions', 'receiving_yards',
    # QB-specific
    'passing_yards', 'completions', 'attempts',
    # RB-specific
    'carries', 'rushing_yards',
]

# Keep only features that actually exist in this position's data
features = [f for f in CANDIDATE_FEATURES if f in df.columns]
print(f"Evaluating {len(features)} features for {POSITION}")

# ── Build feature matrix ───────────────────────────────────────
X = df[features].copy()
y = df['target'].copy()

# Drop rows where target is missing
mask = y.notna()
X, y = X[mask], y[mask]

# Drop columns that are entirely NaN (never populated)
all_nan_cols = X.columns[X.isna().all()].tolist()
if all_nan_cols:
    print(f"Dropping all-NaN features: {all_nan_cols}")
    X = X.drop(columns=all_nan_cols)

# Impute remaining NaNs with column median
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# ── Compute mutual information ─────────────────────────────────
print(f"Computing mutual information scores...")
mi_scores = mutual_info_regression(X_imputed, y, random_state=42)

# ── Build results dataframe ────────────────────────────────────
results = pd.DataFrame({
    'feature': X.columns.tolist(),
    'mutual_info': mi_scores
}).sort_values('mutual_info', ascending=False)

print(f"\nTop features for {POSITION} fantasy points (PPR):\n")
print(results.to_string(index=False))

# ── Correlation with target ────────────────────────────────────
correlations = X.corrwith(y).rename('pearson_corr').reset_index()
correlations.columns = ['feature', 'pearson_corr']
results = results.merge(correlations, on='feature')

# ── Save results ───────────────────────────────────────────────
results.to_csv(f"data/processed/{POSITION.lower()}_feature_scores.csv", index=False)

# ── Plot ───────────────────────────────────────────────────────
top_n = min(15, len(results))
top = results.head(top_n)

fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top['feature'][::-1], top['mutual_info'][::-1])
ax.set_xlabel('Mutual Information Score')
ax.set_title(f'Top {top_n} Features by Information Gain — {POSITION}')
plt.tight_layout()
plt.savefig(f"data/processed/{POSITION.lower()}_feature_importance.png", dpi=150)
plt.show()
print(f"\nPlot saved.")