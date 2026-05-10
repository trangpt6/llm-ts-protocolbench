import pandas as pd
import re
import json
import ast
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
CHAT_FOLDER = BASE_DIR / "logs" / "chat-logs" / "part1-llm-strategic-consultation"
MASTER_LOG = BASE_DIR / "logs" / "master-logs" / "master-log-part1-llm-strategic-consultation.csv"

# Dataset target mapping
DATASET_TARGET_MAP = {
    "AirPassengers":  {"correct_target": "Passengers",                 "dataset_type": "Univariate"},
    "IceCreamHeater": {"correct_target": "Ice cream",                  "dataset_type": "Multivariate"},
    "ILINet":         {"correct_target": "% WEIGHTED ILI",             "dataset_type": "Multivariate"},
    "Temperature":    {"correct_target": "Daily minimum temperatures", "dataset_type": "Univariate"},
    "ETTh1":          {"correct_target": "OT",                         "dataset_type": "Multivariate"},
}

# Model category mapping
MODEL_CATEGORY_MAP = {
    "arima":                "Statistical",
    "sarima":               "Statistical",
    "exponentialsmoothing": "Statistical",
    "holtwinters":          "Statistical",
    "holt":                 "Statistical",
    "ets":                  "Statistical",
    "theta":                "Statistical",
    "tbats":                "Statistical",
    "bats":                 "Statistical",
    "stl":                  "Statistical",
    "naive":                "Statistical",
    "snaive":               "Statistical",
    "prophet":              "Statistical",
    "var":                  "Multivariate Statistical",
    "varma":                "Multivariate Statistical",
    "varmax":               "Multivariate Statistical",
    "vecm":                 "Multivariate Statistical",
    "arimax":               "Multivariate Statistical",
    "sarimax":              "Multivariate Statistical",
    "xgboost":              "ML",
    "xgb":                  "ML",
    "lightgbm":             "ML",
    "lgbm":                 "ML",
    "randomforest":         "ML",
    "svr":                  "ML",
    "elasticnet":           "ML",
    "ridge":                "ML",
    "lasso":                "ML",
    "linearregression":     "ML",
    "lstm":                 "Deep Learning",
    "gru":                  "Deep Learning",
    "transformer":          "Deep Learning",
    "tcn":                  "Deep Learning",
    "nbeats":               "Deep Learning",
    "tft":                  "Deep Learning",
    "deepar":               "Deep Learning",
    "rnn":                  "Deep Learning",
}

def get_model_category(model_name: str) -> str:
    if not model_name:
        return ""
    key = model_name.lower().replace("-", "").replace("_", "").replace(" ", "")
    if key in MODEL_CATEGORY_MAP:
        return MODEL_CATEGORY_MAP[key]
    for k, v in MODEL_CATEGORY_MAP.items():
        if k in key:
            return v
    return "Unknown"

def parse_hyperparameters(hyper_str: str) -> str:
    """Convert Python dict string to JSON, converting tuples to lists."""
    try:
        parsed = ast.literal_eval(hyper_str)

        def convert(obj):
            if isinstance(obj, tuple):
                return list(obj)
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj

        return json.dumps(convert(parsed), ensure_ascii=False)
    except Exception:
        return hyper_str.strip()

def extract_first_llm_response(text: str) -> str:
    """
    Extract the first LLM response block from the chat log.
    Finds the marker 'chatgpt response' / 'claude response' / etc. on its own line.
    """
    response_pattern = re.compile(
        r"^\s*(chatgpt|claude|llm|assistant|gemini|grok|deepseek|kimi|perplexity)\s+response\s*$",
        re.IGNORECASE | re.MULTILINE
    )
    next_section_pattern = re.compile(
        r"^\s*(you asked|chatgpt|claude|llm|assistant|gemini|grok|deepseek|kimi|perplexity)\s*(response|asked)?\s*$",
        re.IGNORECASE | re.MULTILINE
    )

    matches = list(response_pattern.finditer(text))
    if not matches:
        return text  # fallback: toan bo text

    first_match = matches[0]
    start = first_match.end()

    next_match = next_section_pattern.search(text, start)
    end = next_match.start() if next_match else len(text)

    return text[start:end].strip()

def extract_structured_block(response_text: str) -> str:
    """
    From the response text, extract the section starting at 'Data Characteristics Summary:'
    to avoid misparsing free-form analysis written by the LLM before the structured block.
    """
    summary_match = re.search(
        r"Data Characteristics Summary\s*:", response_text, re.IGNORECASE
    )
    if summary_match:
        return response_text[summary_match.start():]
    # Fallback: return full text if no structured header found
    return response_text

# Parse function
def parse_llm_response(text: str, prompt_ver: str) -> tuple:
    errors = []

    data = {
        "Primary_Target":                "",
        "Correct_Target":                False,
        "Data_Characteristics_Trend":    "",
        "Data_Characteristics_Seasonality": "",
        "Data_Characteristics_Noise":    "",
        "Model_Recommended":             "",
        "Model_Category":                "",
        "Hyperparameters_JSON":          "",
        "Justification":                 "",
        "Parsing_Success":               False,
        "Error_Flag":                    False,
    }

    try:
        response_text = extract_first_llm_response(text)
        structured = extract_structured_block(response_text)

# PRIMARY TARGET (v3 only)
        if prompt_ver == "v3":
            # Parse "Primary target column:" only (singular)
            target_match = re.search(
                r"[-*]?\s*Primary target column\s*:\s*(.+)", structured, re.IGNORECASE
            )
            if target_match:
                raw = target_match.group(1).strip().strip("[]'\" ")
                data["Primary_Target"] = raw
                if "," in raw:
                    errors.append("LLM returned multiple targets - expected exactly 1")
            else:
                errors.append("Primary target column field not found in v3 response")
        else:
            data["Primary_Target"] = ""
# TREND / SEASONALITY / NOISE
        trend_match  = re.search(r"[-*]?\s*Trend\s*:\s*(.+)",       structured, re.IGNORECASE)
        season_match = re.search(r"[-*]?\s*Seasonality\s*:\s*(.+)", structured, re.IGNORECASE)
        noise_match  = re.search(r"[-*]?\s*Noise\s*:\s*(.+)",       structured, re.IGNORECASE)

        data["Data_Characteristics_Trend"]       = trend_match.group(1).strip()  if trend_match  else ""
        data["Data_Characteristics_Seasonality"] = season_match.group(1).strip() if season_match else ""
        data["Data_Characteristics_Noise"]       = noise_match.group(1).strip()  if noise_match  else ""

        if not trend_match:
            errors.append("Trend field not found")
        if not season_match:
            errors.append("Seasonality field not found")
        if not noise_match:
            errors.append("Noise field not found")
# RECOMMENDED MODEL
        model_match = re.search(r"Recommended Model\s*:\s*(.+)", structured, re.IGNORECASE)
        if model_match:
            model_name = model_match.group(1).strip()
            data["Model_Recommended"] = model_name
            data["Model_Category"]    = get_model_category(model_name)
        else:
            errors.append("Recommended Model field not found")
# HYPERPARAMETERS
        hyper_match = re.search(
            r"Hyperparameters\s*:\s*(\{.*?\})", structured, re.DOTALL | re.IGNORECASE
        )
        if hyper_match:
            data["Hyperparameters_JSON"] = parse_hyperparameters(hyper_match.group(1).strip())
        else:
            errors.append("Hyperparameters field not found or not a valid dict")
# JUSTIFICATION
        just_match = re.search(
            r"Justification\s*:\s*(.+?)(?=\n\n|\Z)", structured, re.DOTALL | re.IGNORECASE
        )
        if just_match:
            data["Justification"] = just_match.group(1).strip()
        else:
            errors.append("Justification field not found")

        data["Parsing_Success"] = True

    except Exception as e:
        errors.append(f"PARSE EXCEPTION: {str(e)}")

    data["Error_Flag"] = len(errors) > 0
    return data, errors

# Correct target check
def check_correct_target(primary_target: str, correct_target: str, prompt_ver: str) -> bool:
    if prompt_ver == "v2":
        return False
    if not primary_target or not correct_target:
        return False
    if "," in primary_target:
        return False
    return primary_target.strip().lower() == correct_target.strip().lower()

# Main
CHAT_FOLDER.mkdir(parents=True, exist_ok=True)
MASTER_LOG.parent.mkdir(parents=True, exist_ok=True)

records = []

for txt_file in sorted(CHAT_FOLDER.glob("*.txt")):
    with open(txt_file, "r", encoding="utf-8") as f:
        content = f.read()

    stem  = txt_file.stem
    match = re.match(
        r"Part1-(?P<dataset>.+?)-v(?P<prompt_ver>\d+)-S(?P<scenario>\d+)-(?P<llm>.+?)-Run(?P<run_id>\d+)$",
        stem
    )

    if not match:
        print(f"WARNING: Filename does not match expected format, skipping: {txt_file.name}")
        continue

    info       = match.groupdict()
    dataset    = info["dataset"]
    prompt_ver = f"v{info['prompt_ver']}"
    scenario   = f"S{info['scenario']}"
    llm_ver    = info["llm"]
    run_id     = int(info["run_id"])

    dataset_info   = DATASET_TARGET_MAP.get(dataset, {})
    correct_target = dataset_info.get("correct_target", "")
    dataset_type   = dataset_info.get("dataset_type", "Unknown")

    if not dataset_info:
        print(f"WARNING: Dataset '{dataset}' not found in mapping: {txt_file.name}")

    parsed, errors = parse_llm_response(content, prompt_ver)
    parsed["Correct_Target"] = check_correct_target(
        parsed["Primary_Target"], correct_target, prompt_ver
    )

    if errors:
        print(f"  [{txt_file.name}] Errors: {' | '.join(errors)}")

    record = {
        "experiment_part":               "part1-llm-strategic-consultation",
        "dataset":                       dataset,
        "dataset_type":                  dataset_type,
        "scenario":                      scenario,
        "llm_version":                   llm_ver,
        "run_id":                        run_id,
        "prompt_version":                prompt_ver,
        "primary_target":                parsed["Primary_Target"],
        "correct_target":                parsed["Correct_Target"],
        "data_characteristics_trend":    parsed["Data_Characteristics_Trend"],
        "data_characteristics_seasonality": parsed["Data_Characteristics_Seasonality"],
        "data_characteristics_noise":    parsed["Data_Characteristics_Noise"],
        "model_recommended":             parsed["Model_Recommended"],
        "model_category":                parsed["Model_Category"],
        "hyperparameters_json":          parsed["Hyperparameters_JSON"],
        "justification":                 parsed["Justification"],
        "parsing_success":               parsed["Parsing_Success"],
        "error_flag":                    parsed["Error_Flag"],
        "chat_log_file":                 txt_file.name,
        "chat_log_path":                 "llm-ts-protocolbench/" + str(txt_file.relative_to(BASE_DIR)).replace("\\", "/"),
    }
    records.append(record)

new_df = pd.DataFrame(records)

if new_df.empty:
    print("No valid files found to process.")
else:
    if MASTER_LOG.exists():
        existing_df    = pd.read_csv(MASTER_LOG)
        existing_files = set(existing_df["chat_log_file"])
        new_df         = new_df[~new_df["chat_log_file"].isin(existing_files)]

        if not new_df.empty:
            final_df = pd.concat([existing_df, new_df], ignore_index=True)
            print(f"Appending {len(new_df)} new record(s) to master log.")
        else:
            final_df = existing_df
            print("No new records to append.")
    else:
        final_df = new_df
        print(f"Creating master log with {len(final_df)} record(s).")

    # Sort so that higher prompt versions (v3) appear first, lower versions (v2) below.
    if "prompt_version" in final_df.columns:
        final_df["prompt_version_num"] = final_df["prompt_version"].str.extract(r"v(\d+)", expand=False).astype(float)
        final_df = final_df.sort_values(
            by=["prompt_version_num", "dataset", "scenario", "llm_version", "run_id"],
            ascending=[False, True, True, True, True],
            ignore_index=True,
        )
        final_df = final_df.drop(columns=["prompt_version_num"])

    final_df.to_csv(MASTER_LOG, index=False, encoding="utf-8")
    print(f"Done. Total records: {len(final_df)}")
