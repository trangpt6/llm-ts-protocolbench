from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .paths import DATA_DIR, PROMPT_DIR


TEMPERATURE = 0
DEFAULT_MAX_TOKENS = 4096
DEFAULT_NUM_RUNS = 3
DEFAULT_DELAY_BETWEEN_TURNS = 2.0
DEFAULT_DELAY_BETWEEN_RUNS = 3.0


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
