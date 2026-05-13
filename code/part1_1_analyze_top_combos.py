"""
Analyze Part 1 master_log_Recommendation.csv
Goal: For each Dataset × Scenario, find the TOP-1 combo per model category:
        - Statistical        (SARIMA, ExponentialSmoothing, Prophet, ...)
        - Multivariate Statistical (SARIMAX, ...)
        - Machine Learning   (XGBoost, RandomForest, ... - if any)
        - Deep Learning      (LSTM, Transformer, ... - if any)
      If a category has no recommendations in any run, record "N/A (not recommended)"

Output:
  - Console report
  - top_combos_by_category_v3.csv
"""

import json
import pandas as pd
from collections import Counter
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_CSV = BASE_DIR / "logs" / "master-logs" / "master-log-part1-llm-strategic-consultation.csv"
OUTPUT_CSV = BASE_DIR / "results" / "part1-llm-strategic-consultation" / "top-combos-by-category-v3.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Master log not found: {INPUT_CSV}\n"
        "Run part1_0_parse_recommendation_logs.py first to generate it."
    )

CATEGORY_MAP = {
    "Statistical":               "Statistical",
    "Multivariate Statistical":  "Multivariate Statistical",
    "Machine Learning":          "Machine Learning",
    "Deep Learning":             "Deep Learning",
}
CATEGORY_ORDER = [
    "Statistical",
    "Multivariate Statistical",
    "Machine Learning",
    "Deep Learning",
]

df = pd.read_csv(INPUT_CSV)

df_v3 = df[
    (df["prompt_version"] == "v3") &
    (df["parsing_success"].astype(str) == "True")
].copy()

print(f"Total v3 rows (Parsing_Success=True): {len(df_v3)}")
print(f"Datasets  : {sorted(df_v3['dataset'].unique())}")
print(f"Scenarios : {sorted(df_v3['scenario'].unique())}")
print(f"Categories: {sorted(df_v3['model_category'].unique())}")
print()

def normalize_hyperparams(raw):
    if not isinstance(raw, str) or raw.strip() in ("", "{}", "null"):
        return "{}"
    try:
        parsed = json.loads(raw)
        def round_floats(obj):
            if isinstance(obj, dict):  return {k: round_floats(v) for k, v in obj.items()}
            if isinstance(obj, list):  return [round_floats(v) for v in obj]
            if isinstance(obj, float): return round(obj, 4)
            return obj
        return json.dumps(round_floats(parsed), sort_keys=True, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return raw.strip()

def core_hp(hp_norm):
    try:
        obj = json.loads(hp_norm)
    except Exception:
        return hp_norm

    if 'order' in obj and 'seasonal_order' in obj:
        return json.dumps({
            'order': obj['order'],
            'seasonal_order': obj['seasonal_order'],
        }, sort_keys=True)

    if 'seasonal' in obj and 'trend' in obj:
        core = {k: obj[k] for k in ('trend', 'seasonal', 'seasonal_periods') if k in obj}
        return json.dumps(core, sort_keys=True)

    if 'seasonality_mode' in obj:
        core = {k: obj[k] for k in (
            'seasonality_mode', 'yearly_seasonality',
            'weekly_seasonality', 'daily_seasonality'
        ) if k in obj}
        return json.dumps(core, sort_keys=True)

    return hp_norm

def pick_hp(model_grp):
    hp_counts = Counter(model_grp['HP_Norm'])
    top_hp, top_exact_votes = hp_counts.most_common(1)[0]
    if top_exact_votes >= 3:
        return top_hp, top_exact_votes, "exact"

    model_grp = model_grp.copy()
    model_grp['HP_Core'] = model_grp['HP_Norm'].apply(core_hp)
    core_counts = Counter(model_grp['HP_Core'])
    top_core, core_votes = core_counts.most_common(1)[0]
    best_full = Counter(
        model_grp.loc[model_grp['HP_Core'] == top_core, 'HP_Norm']
    ).most_common(1)[0][0]
    return best_full, core_votes, "core-relaxed"

all_results = []
DIVIDER = "=" * 80

for (dataset, scenario), grp in df_v3.groupby(["dataset", "scenario"]):
    total_runs = len(grp)
    print(DIVIDER)
    print(f"  Dataset: {dataset}   |   Scenario: {scenario}   |   Total runs: {total_runs}")
    print(DIVIDER)

    for cat in CATEGORY_ORDER:
        cat_grp = grp[grp["model_category"] == cat]
        cat_count_total = len(cat_grp)

        if cat_count_total == 0:
            print(f"  [{cat}]  -> not recommended in any run (N/A)")
            all_results.append({
                "dataset":             dataset,
                "scenario":            scenario,
                "total_runs":          total_runs,
                "category":            cat,
                "category_total_runs": 0,
                "category_pct":        0.0,
                "model":               "N/A",
                "hyperparameters":     "N/A",
                "combo_count":         0,
                "combo_pct":           0.0,
                "status":              "Not recommended in any run",
            })
            continue

        top_model, model_votes = Counter(cat_grp["model_recommended"]).most_common(1)[0]
        model_grp = cat_grp[cat_grp["model_recommended"] == top_model].copy()
        model_grp["HP_Norm"] = model_grp["hyperparameters_json"].apply(normalize_hyperparams)
        best_hp, best_count, _ = pick_hp(model_grp)
        cat_pct = cat_count_total / total_runs * 100
        combo_pct = best_count / total_runs * 100

        try:
            hp_display = json.dumps(json.loads(best_hp), indent=6, ensure_ascii=False)
        except Exception:
            hp_display = best_hp

        print(f"  [{cat}]  ({cat_count_total} runs = {cat_pct:.1f}% chose this category)")
        print(f"    Best model: {top_model} ({model_votes} runs within category)")
        print(f"    Best combo: {best_count} run(s) ({combo_pct:.1f}% of total)")
        print(f"    Model : {top_model}")
        print(f"    Params: {hp_display}")

        all_results.append({
            "dataset":             dataset,
            "scenario":            scenario,
            "total_runs":          total_runs,
            "category":            cat,
            "category_total_runs": cat_count_total,
            "category_pct":        round(cat_pct, 2),
            "model":               top_model,
            "hyperparameters":     best_hp,
            "combo_count":         best_count,
            "combo_pct":           round(combo_pct, 2),
            "status":              "OK",
        })
    print()

df_out = pd.DataFrame(all_results)
df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8-sig")
print(f"\nSaved: {OUTPUT_CSV}")

print("\nCategory Presence (% runs recommending each category) per Dataset × Scenario")
pivot_pct = (
    df_out[df_out["status"] == "OK"]
    .pivot_table(
        index=["dataset", "category"],
        columns="scenario",
        values="category_pct",
        aggfunc="first",
        fill_value=0,
    )
)
print(pivot_pct.to_string())

print("\nBest Model per Category (collapsed across Scenarios)")
for cat in CATEGORY_ORDER:
    sub = df_out[(df_out["category"] == cat) & (df_out["status"] == "OK")]
    if sub.empty:
        print(f"  {cat}: no data")
        continue
    top = sub.groupby("model")["combo_count"].sum().sort_values(ascending=False)
    print(f"  {cat}: {', '.join([f'{m} ({c})' for m, c in top.items()])}")
