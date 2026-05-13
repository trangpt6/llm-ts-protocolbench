from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_DIR, PROMPT_DIR


TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_NUM_RUNS = 3
DEFAULT_DELAY_BETWEEN_TURNS = 2.0
DEFAULT_DELAY_BETWEEN_RUNS = 3.0

# ---------------------------------------------------------------------------
# Turn-level max_tokens policy
# Turn 3 is computed dynamically: base + expected_length * per_item, capped.
# ---------------------------------------------------------------------------
TURN_MAX_TOKENS_DEFAULTS: dict[int, int] = {0: 4096, 1: 2048, 2: 2048}
TURN_3_MAX_TOKENS_BASE: int = 2400
TURN_3_MAX_TOKENS_PER_ITEM: int = 20
TURN_3_MAX_TOKENS_CAP: int = 8192

# Turn-level request timeouts in seconds (0 = rely on SDK default).
TURN_TIMEOUT_SECONDS: dict[int, int] = {0: 120, 1: 120, 2: 120, 3: 300}

# Maximum characters allowed in a Turn-3 script response.
MAX_SCRIPT_CHARS: int = 16_000

# ---------------------------------------------------------------------------
# Status taxonomy
# ---------------------------------------------------------------------------
# Successful forecasts – the ONLY statuses counted as benchmark checkpoints.
SUCCESS_STATUSES: frozenset[str] = frozenset({"OK_FORECAST_LIST", "OK_FORECAST_SCRIPT"})

# Model-capability failures – no retry, counted as model inability.
MODEL_FAILURE_STATUSES: frozenset[str] = frozenset({
    "MODEL_ERROR_LITERAL",
    "INVALID_LIST_LENGTH",
    "INVALID_LIST_FORMAT",
    "INVALID_SCRIPT_TOO_LONG",
    "INVALID_SCRIPT_FORMAT",
    "INVALID_OUTPUT",
})

# System / API / network failures – logged but separate from model evaluation.
SYSTEM_FAILURE_STATUSES: frozenset[str] = frozenset({
    "SYSTEM_TIMEOUT",
    "SYSTEM_RATE_LIMIT",
    "SYSTEM_API_ERROR",
    "SYSTEM_ALL_KEYS_EXHAUSTED",
    "SYSTEM_REQUEST_FAILED",
    "SYSTEM_EMPTY_RESPONSE",
})

# Legacy statuses from old run logs – treated as success only for skip-checkpoint
# backward compatibility.  New runs NEVER write these.
LEGACY_SUCCESS_STATUSES: frozenset[str] = frozenset({"CHAT_DONE", "OK"})


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    local_path: Path | None
    date_col: str | None
    darts_name: str | None = None


DATASET_SPECS: dict[str, DatasetSpec] = {
    "AirPassengers": DatasetSpec("AirPassengers", DATA_DIR / "AirPassengers.csv", "Month", "AirPassengers"),
    "ETTh1": DatasetSpec("ETTh1", DATA_DIR / "ETTh1.csv", "date", "ETTh1"),
    "ILINet": DatasetSpec("ILINet", DATA_DIR / "ILINet.csv", "DATE", "ILINet"),
    "IceCreamHeater": DatasetSpec("IceCreamHeater", DATA_DIR / "IceCreamHeater.csv", "Month", "IceCreamHeater"),
    "Temperature": DatasetSpec("Temperature", DATA_DIR / "Temperature.csv", "Date", "Temperature"),
}


SCENARIO_DIRS = {
    "S1": PROMPT_DIR / "scenario1_one_step_static",
    "S2": PROMPT_DIR / "scenario2_one_step_rolling",
    "S3": PROMPT_DIR / "scenario3_multi_step_rolling",
    "S4": PROMPT_DIR / "scenario4_block_wise_rolling",
}


MODEL_TRACKS = {
    "baseline": ("baseline_model", "baseline_hyperparameters"),
    "challenger_ml": ("challenger_model_ml", "challenger_hyperparameters_ml"),
    "challenger_dl": ("challenger_model_dl", "challenger_hyperparameters_dl"),
}


MODEL_TRACK_LABELS = {
    "baseline": "Base",
    "challenger_ml": "ChalML",
    "challenger_dl": "ChalDL",
}


DEFAULT_PROVIDER_ORDER = ["openai", "anthropic", "xai", "google", "deepseek", "moonshot"]
