"""
Part 1 -> Part 2 Baseline Extraction

Logic:
  1. Filter Prompt_Version == v3, Parsing_Success == True
  2. Per Dataset × Scenario:
       a. Vote at MODEL level -> pick top-1 model
       b. Among runs that chose that model, vote at HP level:
            - First try exact HP match
            - If top exact HP has < 3 votes, fall back to core-param grouping
              (strip "trend" for SARIMA, "damped_trend" for ETS, prior scales for Prophet)
              then pick the most common full HP within the winning core group
  3. Output: 20-row CSV (5 datasets × 4 scenarios) with 2 key columns:
       Baseline_Model, Baseline_Hyperparameters
     plus supporting vote stats for transparency

Usage:
  python code/part1_analysis_baseline.py
  (reads logs/master-logs/master-log-part1-llm-strategic-consultation.csv)
"""

import json
import pandas as pd
from collections import Counter
from pathlib import Path

# Config
BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "logs" / "master-logs" / "master-log-part1-llm-strategic-consultation.csv"
OUTPUT_CSV = BASE_DIR / "results" / "part1-llm-strategic-consultation" / "part2-model-setup.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Helpers
def normalize_hp(raw):
    """Canonical JSON string: parse -> sort keys -> re-serialize."""
    if not isinstance(raw, str) or raw.strip() in ("", "{}", "null"):
        return "{}"
    try:
        obj = json.loads(raw)
        def rf(o):
            if isinstance(o, dict):  return {k: rf(v) for k, v in o.items()}
            if isinstance(o, list):  return [rf(v) for v in o]
            if isinstance(o, float): return round(o, 4)
            return o
        return json.dumps(rf(obj), sort_keys=True)
    except Exception:
        return raw.strip()

def core_hp(hp_norm):
    """
    Relaxed HP key: strip secondary tuning params that LLMs vary inconsistently.
      SARIMA/SARIMAX  -> keep order, seasonal_order  (drop trend)
      Exp.Smoothing   -> keep trend, seasonal, seasonal_periods  (drop damped_trend)
      Prophet         -> keep seasonality_mode, yearly/weekly/daily_seasonality
                         (drop changepoint_prior_scale, etc.)
      Others          -> return as-is
    """
    try:
        obj = json.loads(hp_norm)
    except Exception:
        return hp_norm

    # SARIMA / SARIMAX
    if 'order' in obj and 'seasonal_order' in obj:
        return json.dumps({
            'order':          obj['order'],
            'seasonal_order': obj['seasonal_order'],
        }, sort_keys=True)

    # Exponential Smoothing
    if 'seasonal' in obj and 'trend' in obj:
        core = {k: obj[k] for k in ('trend', 'seasonal', 'seasonal_periods')
                if k in obj}
        return json.dumps(core, sort_keys=True)

    # Prophet
    if 'seasonality_mode' in obj:
        core = {k: obj[k] for k in
                ('seasonality_mode', 'yearly_seasonality',
                 'weekly_seasonality', 'daily_seasonality')
                if k in obj}
        return json.dumps(core, sort_keys=True)

    return hp_norm

def pick_hp(model_grp):
    """
    Given a DataFrame of runs that all chose the same model,
    return the best representative HP string.
    Strategy: exact vote first; fall back to core-group vote if top exact < 3.
    """
    hp_counts = Counter(model_grp['HP_Norm'])
    top_hp, top_exact_votes = hp_counts.most_common(1)[0]

    if top_exact_votes >= 3:
        return top_hp, top_exact_votes, "exact"

    # Fall back: group by core HP, pick largest core group,
    # then return the most common full HP within that group
    model_grp = model_grp.copy()
    model_grp['HP_Core'] = model_grp['HP_Norm'].apply(core_hp)
    core_counts = Counter(model_grp['HP_Core'])
    top_core, core_votes = core_counts.most_common(1)[0]
    best_full = Counter(
        model_grp.loc[model_grp['HP_Core'] == top_core, 'HP_Norm']
    ).most_common(1)[0][0]
    return best_full, core_votes, "core-relaxed"

# Load and filter
df = pd.read_csv(INPUT_CSV)
df_v3 = df[
    (df['prompt_version'] == 'v3') &
    (df['parsing_success'].astype(str) == 'True')
].copy()

df_v3['HP_Norm'] = df_v3['hyperparameters_json'].apply(normalize_hp)

print(f"Loaded: {len(df_v3)} runs (v3, parsed successfully)")
print(f"Datasets : {sorted(df_v3['dataset'].unique())}")
print(f"Scenarios: {sorted(df_v3['scenario'].unique())}\n")

# Metadata
SCENARIO_DESC = {
    "S1": "One-step ahead | Static setup | Train once | Recursive inference | Ground truth disabled | No retraining",
    "S2": "One-step ahead | Rolling update | Retrain every step | Ground truth enabled | Continuous retraining",
    "S3": "Multi-step ahead (H) | Rolling update | Retrain every step | Ground truth enabled | Continuous retraining",
    "S4": "Multi-step ahead (B) | Block-wise rolling | Retrain every block | Ground truth enabled | Periodic retraining",
}
DATASET_TYPE = {
    "AirPassengers":  "Univariate",
    "ETTh1":          "Multivariate",
    "IceCreamHeater": "Multivariate",
    "ILINet":         "Multivariate",
    "Temperature":    "Univariate",
}
DATASET_PRIMARY_TARGET = {
    "AirPassengers":  "Passengers",
    "ETTh1":          "OT",
    "IceCreamHeater": "Ice cream",
    "ILINet":         "% WEIGHTED ILI",
    "Temperature":    "Daily minimum temperatures",
}

# Main loop
rows = []
print(f"{'Dataset':<16} {'Sc':<4} {'Model':<22} {'Model%':>7} {'HP%':>6}  {'HP_Method':<14} Hyperparameters")
print("-" * 110)

for (ds, sc), grp in df_v3.groupby(['dataset', 'scenario']):
    total = len(grp)
    model_counts = Counter(grp['model_recommended'])
    top_model, model_votes = model_counts.most_common(1)[0]
    model_pct = model_votes / total * 100

    model_grp = grp[grp['model_recommended'] == top_model]
    best_hp, hp_votes, method = pick_hp(model_grp)
    hp_pct = hp_votes / total * 100
    horizon_blocksize = 1 if sc in ("S1", "S2") else ""

    print(f"{ds:<16} {sc:<4} {top_model:<22} {model_pct:>6.0f}% {hp_pct:>5.0f}%  {method:<14} {best_hp}")

    rows.append({
        "dataset": ds,
        "dataset_type": DATASET_TYPE.get(ds, ""),
        "dataset_primary_target": DATASET_PRIMARY_TARGET.get(ds, ""),
        "scenario": sc,
        "scenario_description": SCENARIO_DESC.get(sc, ""),
        "horizon_or_blocksize": horizon_blocksize,
        "total_runs": total,
        "baseline_model": top_model,
        "baseline_hyperparameters": best_hp,
        "model_votes": model_votes,
        "model_pct": round(model_pct, 1),
        "hp_votes": hp_votes,
        "hp_pct": round(hp_pct, 1),
        "hp_selection_method": method,
        "challenger_model_ml": "",
        "challenger_hyperparameters_ml": "",
        "challenger_model_dl": "",
        "challenger_hyperparameters_dl": "",
    })

# Save
df_out = pd.DataFrame(rows)
df_out.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')
print(f"\nSaved {len(df_out)} rows to {OUTPUT_CSV}")
print(f"Columns: {list(df_out.columns)}")
