"""
Parse Part 2 chat logs into one structured master CSV row per run.

File naming convention:
    Part2-<Dataset>-<Scenario>-<Branch>-<LLMVersion>-Run<N>.txt
    e.g.  Part2-Temperature-S4-ChalDL-DeepSeekV4-Run1.txt

Current API chat format:
    ===== TURN 0 ASSISTANT (<LLMName>) =====
    ...
    ===== TURN 3 ASSISTANT (<LLMName>) =====

Legacy web-exported header formats are still supported:
    '<name> response'   e.g.  deepseek response
    '<name> said:'      e.g.  ChatGPT said:       (some ChatGPT export formats)
    '<name> said'       e.g.  ChatGPT said

Turn 3 - three content columns in master log:
    T3_Raw_Response   : full text of last LLM response (always filled)
    T3_Script_Content : extracted Python code  (filled when SCRIPT, else "")
    T3_Forecast_List  : extracted list string  (filled when LIST, else "").
                        For SCRIPT rows this stays blank until metrics execution.

Saved files on disk:
    results/part2-interactive-llm-forecasting/scripts-output/<file_id>.py   (when output is SCRIPT)
    results/part2-interactive-llm-forecasting/raw-forecasts/<file_id>.json   (when output is LIST)
"""

import ast
import re
from pathlib import Path

import pandas as pd

from part2_api.model_setup import load_model_setup
from part2_api.paths import DEFAULT_MODEL_SETUP_PATH

# PATH CONFIGURATION
BASE_DIR          = Path(__file__).resolve().parent.parent
CHAT_LOGS_DIR     = BASE_DIR / "logs" / "chat-logs" / "part2-interactive-llm-forecasting"
RAW_OUTPUT_DIR    = BASE_DIR / "results" / "part2-interactive-llm-forecasting" / "raw-forecasts"
SCRIPT_OUTPUT_DIR = BASE_DIR / "results" / "part2-interactive-llm-forecasting" / "scripts-output"
MASTER_LOG_PATH   = BASE_DIR / "logs" / "master-logs" / "master-log-part2-interactive-llm-forecasting.csv"
DATA_DIR          = BASE_DIR / "data"

RAW_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SCRIPT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MASTER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

# Dataset name -> CSV filename mapping
DATASET_CSV_MAP = {
    "AirPassengers": "AirPassengers.csv",
    "ETTh1": "ETTh1.csv",
    "IceCreamHeater": "IceCreamHeater.csv",
    "ILINet": "ILINet.csv",
    "Temperature": "Temperature.csv",
}

_EXPECTED_LENGTH_CACHE: dict[str, int | None] = {}

# All known LLM name keywords that appear before ' response' / ' said:?' in chat exports.
KNOWN_LLM_HEADERS = [
    "deepseek", "chatgpt", "claude", "gemini", "grok", "kimi", "llm",
]


def detect_llm_header(text: str) -> tuple[str, str] | None:
    """
    Scan the file content for the first LLM response header on its own line.

    Supported formats (header must occupy a line by itself):
        '<name> response'   -> style 'response'
        '<name> said:'      -> style 'said'
        '<name> said'       -> style 'said'

    Returns (name, style) e.g. ('chatgpt', 'said') or ('deepseek', 'response').
    Returns None if no known header is found.
    """
    names   = '|'.join(KNOWN_LLM_HEADERS)
    pattern = re.compile(
        rf'^\s*(?P<name>{names})\s+(?P<style>response|said:?)\s*$',
        re.IGNORECASE | re.MULTILINE
    )
    m = pattern.search(text)
    if not m:
        return None
    style = "said" if m.group("style").lower().startswith("said") else "response"
    return (m.group("name").lower(), style)


def split_turns(text: str, llm_header: tuple) -> list[dict]:
    """
    Split the chat log into [{"role": "user"|"llm", "content": str}, ...].

    Boundary rules (same as Part1 extract_first_llm_response):
    - Headers must appear alone on their own line (^\s*<header>\s*$).
      This prevents false matches inside response body text.
    - An LLM turn starts at the detected header line ('<name> response' or '<name> said:?').
    - A user turn starts at a 'you asked' line.
    - Each segment ends when the next boundary line is encountered.

    llm_header is the (name, style) tuple from detect_llm_header().
    """
    name, style = llm_header
    escaped     = re.escape(name)
    llm_pat     = (
        rf'^\s*{escaped}\s+response\s*$'
        if style == "response"
        else rf'^\s*{escaped}\s+said:?\s*$'
    )
    boundary = re.compile(
        rf'(?P<llm>{llm_pat})|(?P<user>^\s*you\s+asked\s*$)',
        re.IGNORECASE | re.MULTILINE
    )

    segments  = []
    last_end  = 0
    last_role = None

    for m in boundary.finditer(text):
        if last_role is not None:
            segments.append({
                "role":    last_role,
                "content": text[last_end:m.start()].strip()
            })
        last_role = "llm" if m.group("llm") else "user"
        last_end  = m.end()

    if last_role is not None:
        segments.append({"role": last_role, "content": text[last_end:].strip()})

    return segments


def get_llm_turns(text: str, llm_header: tuple) -> list[str]:
    """Return only the LLM response strings in chronological order."""
    return [t["content"] for t in split_turns(text, llm_header) if t["role"] == "llm"]


def get_api_chat_turns(text: str) -> dict[int, str]:
    """
    Parse the local API chat format generated by code/run_part2_api.py.

    The API runner stores one assistant section per turn:
        ===== TURN 0 ASSISTANT (<LLMName>) =====
        ...
    """
    pattern = re.compile(
        r"^===== TURN\s+(?P<turn>\d+)\s+ASSISTANT[^\n]*=====\s*\n"
        r"(?P<content>.*?)(?=^===== TURN\s+\d+\s+(?:USER|ASSISTANT)[^\n]*=====|\Z)",
        re.IGNORECASE | re.MULTILINE | re.DOTALL,
    )
    return {int(m.group("turn")): m.group("content").strip() for m in pattern.finditer(text)}


def classify_last_response(response: str) -> tuple[str, str, str]:
    """
    Classify the last LLM response and extract structured content.

    Returns
    -------
    output_type    : "SCRIPT" | "LIST" | "ERROR" | "TEXT"
    script_content : clean Python code string (or "")
    forecast_list  : Python list literal string (or "")

    Detection order
    ---------------
    1. Bare "ERROR" -> ERROR
    2. Contains Python code lines -> SCRIPT; extract from first code line onward.
    3. Contains a numeric Python list [...] -> LIST
    4. Otherwise -> TEXT (unstructured, could not parse)
    """
    stripped = response.strip()

    if stripped.upper() == "ERROR":
        return "ERROR", "", ""

    script_content = extract_script_content(stripped)
    if script_content:
        return "SCRIPT", script_content, ""

    forecast_list = extract_forecast_list_literal(stripped)
    if forecast_list:
        return "LIST", "", forecast_list

    return "TEXT", "", ""


def extract_script_content(text: str) -> str:
    """
    Extract executable Python from the first real code line onward.

    Some LLMs wrap the whole script in triple quotes.  If the wrapper appears
    before the first real code line, drop only that outer wrapper while leaving
    legitimate inner docstrings untouched.
    """
    code_start = re.search(
        r'^(import\s+\w+|from\s+\w+\s+import|'
        r'class\s+\w+\s*[\(:]|def\s+\w+\s*\(|'
        r'(?:random|np|torch|pd|plt)\.\w+\s*\(|'
        r'(?:np|torch)\.random\.\w+)',
        text,
        re.MULTILINE,
    )
    if not code_start:
        return ""

    prefix = text[: code_start.start()]
    script = text[code_start.start() :].strip()
    wrapper = detect_outer_script_wrapper(prefix)
    if wrapper:
        closing_idx = script.rfind(wrapper)
        if closing_idx != -1 and not script[closing_idx + len(wrapper) :].strip():
            script = script[:closing_idx].rstrip()
    return strip_trailing_markdown_fences(script)


def strip_trailing_markdown_fences(script: str) -> str:
    """Drop closing markdown fences that can appear after extracted code."""
    lines = script.rstrip().splitlines()
    while lines and re.fullmatch(r"\s*```\s*", lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def detect_outer_script_wrapper(prefix: str) -> str:
    """
    Return an outer triple-quote delimiter when only wrapper text precedes code.

    The prefix may contain whitespace and one optional language hint such as
    `python`, but no other prose.  That keeps normal explanations from being
    mistaken for wrappers.
    """
    cleaned = prefix.strip()
    for delimiter in ("'''", '"""'):
        if cleaned == delimiter:
            return delimiter
        if cleaned.lower() in {f"python{delimiter}", f"{delimiter}python"}:
            return delimiter
    return ""


def extract_forecast_list_literal(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("["):
        candidate = stripped[: stripped.rfind("]") + 1] if "]" in stripped else stripped
        try:
            value = ast.literal_eval(candidate)
            if isinstance(value, list):
                return candidate
        except Exception:
            pass

    for match in re.finditer(r"\[[^\[\]]+\]", stripped, flags=re.DOTALL):
        candidate = match.group(0).strip()
        try:
            value = ast.literal_eval(candidate)
        except Exception:
            continue
        if isinstance(value, list) and all(not isinstance(v, (list, tuple, dict)) for v in value):
            return candidate
    return ""


def _record_value(record: dict, *names: str):
    """Fetch a value by exact or case-insensitive column name."""
    for name in names:
        if name in record:
            return record[name]
    lowered = {str(k).lower(): v for k, v in record.items()}
    for name in names:
        key = name.lower()
        if key in lowered:
            return lowered[key]
    return ""


def _has_value(value) -> bool:
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip() != ""


def _normalise_output_type(value) -> str:
    text = str(value or "").strip().lower()
    aliases = {
        "list": "list",
        "forecast_list": "list",
        "script": "script",
        "error": "error",
        "model_error_literal": "error",
        "text": "invalid",
        "invalid": "invalid",
        "invalid_output": "invalid",
        "system_fail": "system",
    }
    return aliases.get(text, text)


def _parse_flat_numeric_list(text: str) -> list[float] | None:
    try:
        value = ast.literal_eval(str(text).strip())
    except Exception:
        return None
    if not isinstance(value, list):
        return None

    parsed = []
    for item in value:
        if isinstance(item, bool) or isinstance(item, (list, tuple, dict)):
            return None
        try:
            parsed.append(float(item))
        except (TypeError, ValueError):
            return None
    return parsed


def _expected_forecast_length_for_dataset(dataset: str) -> int | None:
    dataset = str(dataset or "").strip()
    if not dataset:
        return None
    if dataset in _EXPECTED_LENGTH_CACHE:
        return _EXPECTED_LENGTH_CACHE[dataset]

    csv_name = DATASET_CSV_MAP.get(dataset)
    if not csv_name:
        _EXPECTED_LENGTH_CACHE[dataset] = None
        return None
    try:
        n_rows = len(pd.read_csv(DATA_DIR / csv_name))
    except Exception:
        _EXPECTED_LENGTH_CACHE[dataset] = None
        return None

    split_idx = int(n_rows * 0.8)
    expected = n_rows - split_idx if 0 < split_idx < n_rows else None
    _EXPECTED_LENGTH_CACHE[dataset] = expected
    return expected


def derive_analysis_status(record: dict) -> str:
    """
    Derived historical status used for analysis only.

    The original stored status is left unchanged. Older rows can have
    OK_FORECAST_LIST even when the list length is wrong, so list successes are
    rechecked strictly before analysis.
    """
    original_status = str(_record_value(record, "run_status", "status") or "").strip()
    raw_response = str(_record_value(record, "T3_Raw_Response", "t3_raw_response") or "")
    if raw_response.strip().upper() == "ERROR" or original_status == "MODEL_ERROR_LITERAL":
        return "MODEL_ERROR_LITERAL"

    output_type = _normalise_output_type(_record_value(record, "t3_output_type"))
    expected_length = _expected_forecast_length_for_dataset(_record_value(record, "dataset"))

    # Historical rows without validator length columns are rechecked by parsing
    # the Turn 3 payload as a flat numeric list before any length decision.
    parsed_list = _parse_flat_numeric_list(raw_response)
    if parsed_list is None:
        parsed_list = _parse_flat_numeric_list(
            _record_value(record, "T3_Forecast_List", "t3_forecast_list")
        )
    if parsed_list is not None and expected_length is not None:
        return "OK_FORECAST_LIST" if len(parsed_list) == expected_length else "INVALID_LIST_LENGTH"

    if output_type == "list":
        return original_status if original_status and original_status != "OK_FORECAST_LIST" else "INVALID_OUTPUT"
    if output_type == "script":
        return original_status if original_status else "OK_FORECAST_SCRIPT"
    if output_type == "error":
        return "MODEL_ERROR_LITERAL"
    if output_type == "invalid":
        if original_status.startswith(("INVALID_", "SYSTEM_")) or original_status == "MODEL_ERROR_LITERAL":
            return original_status
        return "INVALID_OUTPUT"
    if original_status:
        return original_status
    return ""


def fix_csv_path(script_content: str, dataset_name: str) -> str:
    """
    Replace any pd.read_csv('...') or pd.read_csv("...") call with the
    actual relative path to the dataset file, e.g.:
        pd.read_csv('AirPassengers.csv')
        -> pd.read_csv(r'../../../data/AirPassengers.csv')
    Keeps any extra keyword arguments (e.g. parse_dates=, dayfirst=) intact.
    """
    csv_filename = DATASET_CSV_MAP.get(dataset_name, f"{dataset_name}.csv")
    actual_path  = DATA_DIR / csv_filename

    try:
        rel_path = actual_path.relative_to(BASE_DIR)
        # scripts-output is 3 levels deep: results/part2-interactive-llm-forecasting/scripts-output/
        rel_str = "../../../" + "/".join(rel_path.parts)
    except ValueError:
        rel_str = str(actual_path)

    def _replace(m):
        return f"pd.read_csv(r'{rel_str}'{m.group(1)})"

    return re.sub(
        r"pd\.read_csv\(\s*['\"][^'\"]*['\"]\s*((?:,\s*[^)]+)?)\)",
        _replace,
        script_content
    )


# TURN 0: data inspection
def parse_turn0(response: str) -> dict:
    def _field(label: str) -> str:
        m = re.search(rf'{re.escape(label)}\s*[:\-]\s*(.+)', response, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    return {
        "t0_rows":                 _field("Rows"),
        "t0_columns":              _field("Columns"),
        "t0_time_index":           _field("Time Index"),
        "t0_primary_target":       _field("Primary Target"),
        "t0_frequency":            _field("Frequency"),
        "t0_missing_values":       _field("Missing Values"),
        "t0_duplicated_timestamps": _field("Duplicated Timestamps"),
        "t0_irregularities":       _field("Raw Data Irregularities"),
        "t0_trend":                _field("Trend"),
        "t0_seasonality":          _field("Seasonality"),
        "t0_noise":                _field("Noise"),
        "t0_split_rule":           _field("Split Rule"),
        "t0_train_size":           _field("Train Size"),
        "t0_test_size":            _field("Test Size"),
        "t0_train_period":         _field("Train Period"),
        "t0_test_period":          _field("Test Period"),
        "t0_format_followed":      bool(re.search(r'Dataset Inspection', response, re.IGNORECASE)),
    }


# TURN 1: preprocessing decision
def parse_turn1(response: str) -> dict:
    def _field(label: str) -> str:
        m = re.search(rf'{re.escape(label)}\s*[:\-]\s*(.+)', response, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    needed_raw = _field("Needed")
    needed     = needed_raw.lower().startswith("yes") if needed_raw else False

    return {
        "t1_preprocessing_needed": needed,
        "t1_issues_detected":      _field("Issues Detected"),
        "t1_preprocessing_steps":  _field("Preprocessing Steps"),
        "t1_zerovalue_treatment":  _field("Treatment of Zero Values"),
        "t1_justification":        _field("Justification"),
        "t1_has_previous_summary": bool(re.search(r'Previous Decisions Summary', response, re.IGNORECASE)),
        "t1_format_followed":      bool(re.search(r'Preprocessing Decision', response, re.IGNORECASE)),
        "t1_overcleaned_flag":     needed and bool(
            re.search(r'(normaliz|standardiz|scal|log.transfor|differenc)', _field("Preprocessing Steps"), re.IGNORECASE)
        ),
    }


# TURN 2: setup understanding
def parse_turn2(response: str) -> dict:
    def _field(label: str) -> str:
        m = re.search(rf'{re.escape(label)}\s*[:\-]\s*(.+)', response, re.IGNORECASE)
        return m.group(1).strip() if m else ""

    understanding = _field("Scenario Understanding")
    key_impl      = _field("Key Implication")

    check = (understanding + " " + key_impl).lower()
    return {
        "t2_scenario_understanding":   understanding,
        "t2_model_confirmed":         _field("Fixed Model"),
        "t2_hyperparams_confirmed":   _field("Fixed Hyperparameters"),
        "t2_key_implication":          key_impl,
        "t2_has_previous_summary":    bool(re.search(r'Previous Decisions Summary', response, re.IGNORECASE)),
        "t2_format_followed":         bool(re.search(r'Setup Understanding', response, re.IGNORECASE)),
        "t2_understands_blockwise":   bool(re.search(
            r'block.{0,30}(?:rolling|window|loop|forecast|predict|train|by.block)|'
            r'(?:sliding|moving).{0,20}block|blockwise|block.wise|'
            r'by.{0,10}block',
            check)),
        "t2_understands_retraining":  bool(re.search(
            r'(?:retrain|re.train|refit|re.fit).{0,30}(?:block|each|every|iteration|after|step|model)|'
            r'(?:train|fit)\s+(?:again|anew|each\s+(?:block|step)|after\s+each)|'
            r'every\s+(?:block|step|iteration).{0,30}(?:retrain|refit|train)',
            check)),
        "t2_understands_groundtruth": bool(re.search(
            r'(?:ground.truth|actual.val|test.label|true.val|observed|real.data).{0,30}(?:extend|append|add|incorporate|fed|concatenat|merge|update)|'
            r'(?:actual|true|observed|real)\s+(?:value|data).{0,30}(?:fed|added|incorporated|included|concatenat)|'
            r'(?:update|augment).{0,20}(?:train|data).{0,20}(?:actual|true|ground|observed)|'
            r'(?:ground|actual|true).{0,30}(?:concatenat|merge|join|combine).{0,20}(?:train|data)',
            check))
        or bool(re.search(r'true.value.{0,30}(?:extend|append|use)', check)),
    }


# TURN 3: last LLM response (forecast / code)

# ---------------------------------------------------------------------------
# Helper: correct block size per dataset for S4 (blockwise scenario).
# Only S4 has a block_size concept; for other scenarios this check is skipped.
# ---------------------------------------------------------------------------
_S4_BLOCK_SIZE_CACHE: dict[str, int] = {}

def _get_s4_block_size(dataset: str, scenario: str) -> int | None:
    """Return the correct block_size for S4, or None when not applicable."""
    scenario_norm = scenario.strip().upper()
    if scenario_norm not in ("S4", "SCENARIO4", "4"):
        return None  # block_size check only applies to S4
    dataset = dataset.strip()
    cache_key = f"{dataset}|S4"
    if cache_key in _S4_BLOCK_SIZE_CACHE:
        return _S4_BLOCK_SIZE_CACHE[cache_key]
    try:
        setup = load_model_setup(DEFAULT_MODEL_SETUP_PATH, dataset, "S4", "baseline")
        _S4_BLOCK_SIZE_CACHE[cache_key] = setup.horizon_or_block
        return setup.horizon_or_block
    except Exception:
        _S4_BLOCK_SIZE_CACHE[cache_key] = None
        return None


def _check_correct_block_size(code: str, dataset: str, scenario: str) -> bool:
    """True when the code uses the correct block_size for this dataset+scenario.

    For S4 we look up the ground-truth block size from the model-setup CSV and
    check that the script assigns it to block_size, PRED_LEN, or pred_len.
    For other scenarios the check is always True (no block concept).
    """
    expected = _get_s4_block_size(dataset, scenario)
    if expected is None:
        return True  # not S4 → nothing to check
    # Build a regex that matches block_size = <expected> (case-insensitive,
    # tolerating whitespace around '=').
    return bool(re.search(
        rf'(?:block_size|PRED_LEN|pred_len)\s*=\s*{expected}\b',
        code, re.IGNORECASE,
    ))


def parse_turn3(
    raw_response:   str,
    output_type:    str,
    script_content: str,
    forecast_list:  str,
    dataset:        str = "",
    scenario:       str = "",
) -> dict:
    """
    Build the Turn 3 section of the master log record.

    T3_Raw_Response: full original text of the last LLM response.
    T3_Script_Content: extracted Python code when the response is SCRIPT.
    T3_Forecast_List: extracted forecast list string when the response is LIST.
                      Stays empty for SCRIPT rows - paste manually after running.
    """
    code = script_content

    forecast_length = None
    if forecast_list:
        try:
            forecast_length = len(ast.literal_eval(forecast_list))
        except Exception:
            pass

    # Extra prose before code: LLM added text/headings before first import
    extra_text_flag = False
    if output_type == "SCRIPT":
        first_import = re.search(r'^(import|from)\s', raw_response.strip(), re.MULTILINE)
        if first_import and first_import.start() > 80:
            extra_text_flag = True

    return {
        "t3_raw_response":           raw_response,
        "t3_script_content":         script_content,
        "t3_forecast_list":          forecast_list,
        "t3_output_type":            output_type,
        "t3_parse_success":          output_type in ("SCRIPT", "LIST"),
        "t3_extra_text_flag":        extra_text_flag,
        "t3_forecast_length":        forecast_length,
        "t3_has_seed":               bool(re.search(r'random\.seed|np\.random\.seed|torch\.manual_seed|set_seed', code)),
        "t3_reads_csv":              bool(re.search(r'read_csv', code)),
        "t3_correct_split_80_20":    bool(re.search(r'0\.8\s*\*|int\(.*0\.8|n_train\s*=\s*\d+|train_size\s*=\s*\d+|n_test\s*=\s*\d+', code)),
        "t3_correct_block_size":     _check_correct_block_size(code, dataset, scenario),
        "t3_retraining_implemented": bool(re.search(
            r'for\s+block|block_idx|train_model\(current_train|'
            r'for\s+\w+\s+in\s+range\s*\([^)]*\)\s*:.*?\.fit\(|'
            r'while\s+[^:]+:.*?\.fit\(',
            code, re.IGNORECASE | re.DOTALL)),
        "t3_uses_ground_truth":      bool(re.search(
            r'(?:extend|append|\.append\(|\.extend\()\s*.*?(?:test_val|test|y_test|X_test|y_true)|'
            r'current_train\s*=\s*np\.concatenate|'
            r'np\.concatenate\s*\(.*?(?:test|y_test|y_true)|'
            r'(?:X_train|y_train)\s*=\s*np\.concat|'
            r'y_train\s*=\s*y_train\s*\+.*?(?:test|y_test)',
            code, re.IGNORECASE | re.DOTALL)),
        "t3_remainder_handled":      bool(re.search(r'remainder|len.*%\s*block|%\s*block_size', code, re.IGNORECASE)),
        "t3_prints_forecasts":       bool(re.search(r'print\(\s*forecasts\s*\)', code)),
        "t3_uses_torch":             bool(re.search(r'import torch', code)),
        "t3_uses_sklearn":           bool(re.search(r'from sklearn|import sklearn', code)),
        "t3_uses_statsmodels":       bool(re.search(r'import statsmodels|from statsmodels', code)),
        "t3_uses_xgboost":           bool(re.search(r'import xgboost|XGBRegressor', code)),
        "t3_uses_prophet":           bool(re.search(r'from prophet|import prophet', code)),
    }


# File metadata parsed from filename stem
def parse_filename(stem: str) -> dict:
    """
    Expected: Part2-<Dataset>-<Scenario>-<Branch>-<LLMVersion>-Run<N>
    Falls back to 'NA' for missing segments.
    """
    parts  = stem.split('-')
    run_id = ""
    if len(parts) >= 6:
        run_id = re.sub(r'[Rr]un', '', parts[5]).strip()

    return {
        "experiment_part": parts[0] if len(parts) > 0 else "NA",
        "dataset":         parts[1] if len(parts) > 1 else "NA",
        "scenario":        parts[2] if len(parts) > 2 else "NA",
        "branch":          parts[3] if len(parts) > 3 else "NA",
        "llm_version":     parts[4] if len(parts) > 4 else "NA",
        "run_id":          run_id    if run_id          else "NA",
        "chat_log_file":   stem + ".txt",
    }


def save_output_file(
    file_id:        str,
    output_type:    str,
    script_content: str,
    raw_response:   str,
) -> tuple[str, str]:
    """
    SCRIPT -> scripts-output/<file_id>.py  (saves the clean extracted code)
    LIST   -> raw-forecasts/<file_id>.json (saves full raw response for recovery)
    Returns (script_file_path, raw_file_path) as relative path strings ("" when not saved).
    """
    script_file = raw_file = ""

    if output_type == "SCRIPT" and script_content:
        out = SCRIPT_OUTPUT_DIR / f"{file_id}.py"
        out.write_text(script_content, encoding="utf-8")
        rel_path = out.relative_to(BASE_DIR)
        script_file = str(rel_path).replace("\\", "/")

    elif output_type == "LIST":
        out = RAW_OUTPUT_DIR / f"{file_id}.json"
        out.write_text(raw_response.strip(), encoding="utf-8")
        rel_path = out.relative_to(BASE_DIR)
        raw_file = str(rel_path).replace("\\", "/")

    return script_file, raw_file


# Process one chat log file
def process_file(file_path: Path) -> dict | None:
    file_id = file_path.stem
    print(f"  {file_id}")

    # Extract dataset & scenario from filename early (needed for block-size
    # validation and CSV-path fix below).
    fmeta   = parse_filename(file_id)
    dataset  = fmeta.get("dataset", "")
    scenario = fmeta.get("scenario", "")

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"    [ERROR] Cannot read: {e}")
        return None

    api_turns = get_api_chat_turns(content)
    if api_turns:
        # Count only assistant turns that are present (keys 0-3)
        n_turns = sum(1 for k in api_turns if isinstance(k, int))
        t0_resp = api_turns.get(0, "")
        t1_resp = api_turns.get(1, "")
        t2_resp = api_turns.get(2, "")
        t3_resp = api_turns.get(3, api_turns[max(api_turns)])
        print(f"    format=api | n_turns={n_turns} | "
              f"T0={'yes' if t0_resp else 'no'} "
              f"T1={'yes' if t1_resp else 'no'} "
              f"T2={'yes' if t2_resp else 'no'} "
              f"T3={'yes' if t3_resp else 'no'}")
    else:
        # LLM_Version comes from the filename (e.g. 'GPT53').
        # The split boundary comes from scanning web-exported chat content.
        llm_header = detect_llm_header(content)
        if not llm_header:
            print(f"    [WARN] No LLM response header detected - check file format")
            return None

        llm_turns = get_llm_turns(content, llm_header)
        n_turns   = len(llm_turns)

        if n_turns == 0:
            print(f"    [WARN] No LLM turns found after splitting on '{llm_header}'")
            return None

        def _get(lst, i): return lst[i] if i < len(lst) else ""

        t0_resp = _get(llm_turns, 1)   # data inspection
        t1_resp = _get(llm_turns, 2)   # preprocessing decision
        t2_resp = _get(llm_turns, 3)   # setup understanding
        t3_resp = llm_turns[-1]        # always the last LLM response (forecast / code)

        print(f"    header={llm_header} | n_turns={n_turns} | "
              f"T0={'yes' if t0_resp else 'no'} "
              f"T1={'yes' if t1_resp else 'no'} "
              f"T2={'yes' if t2_resp else 'no'} "
              f"T3(last)={'yes' if t3_resp else 'no'}")

    # Determine Turn-3 output only from the actual Turn-3 response body.
    output_type, script_content, forecast_list = classify_last_response(t3_resp)
    print(f"    T3 type -> {output_type}")

    # Fix read_csv paths before saving the script
    if output_type == "SCRIPT" and script_content:
        script_content = fix_csv_path(script_content, dataset)

    script_file, raw_file = save_output_file(file_id, output_type, script_content, t3_resp)

    t3_record = parse_turn3(t3_resp, output_type, script_content, forecast_list,
                             dataset=dataset, scenario=scenario)
    record = {
        **parse_filename(file_id),
        "n_llm_turns": n_turns,
        "script_file": script_file,
        "raw_file":    raw_file,
        **parse_turn0(t0_resp),
        **parse_turn1(t1_resp),
        **parse_turn2(t2_resp),
        **t3_record,
    }
    record["Analysis_Status"] = derive_analysis_status(record)
    return record


# MAIN
def main():
    if not CHAT_LOGS_DIR.exists():
        print(f"[ERROR] Directory not found: {CHAT_LOGS_DIR}")
        return

    files = sorted(CHAT_LOGS_DIR.glob("*.txt"))
    if not files:
        print(f"[WARN] No .txt files found in {CHAT_LOGS_DIR}")
        return

    print(f"Found {len(files)} file(s) - parsing...\n")

    records, errors = [], []
    for fp in files:
        rec = process_file(fp)
        if rec:
            records.append(rec)
        else:
            errors.append(fp.name)

    if not records:
        print("No records produced.")
        return

    df = pd.DataFrame(records)
    df.to_csv(MASTER_LOG_PATH, index=False, encoding="utf-8-sig")
    print(f"\n{len(records)} record(s) saved -> {MASTER_LOG_PATH}")
    if errors:
        print(f"  Failed files: {errors}")

    print("\nT3 output type breakdown")
    if "t3_output_type" in df.columns:
        print(df["t3_output_type"].value_counts().to_string())
    if "t3_parse_success" in df.columns:
        rate = df["t3_parse_success"].mean() * 100
        print(f"Parse success rate: {rate:.1f}%")

if __name__ == "__main__":
    main()
