from __future__ import annotations

from dataclasses import dataclass

from .datasets import DatasetBundle, csv_for_prompt
from .model_setup import ModelSetup
from .settings import SCENARIO_DIRS


@dataclass(frozen=True)
class PromptTurn:
    turn_id: int
    label: str
    content: str


TURN_FILES = [
    (0, "data_inspection", "turn0_data_inspection.txt"),
    (1, "preprocessing_decision", "turn1_preprocessing_decision.txt"),
    (2, "setup_understanding", "turn2_fixed_setup_and_understanding.txt"),
    (3, "train_and_forecast", "turn3_train_and_forecast.txt"),
]


def build_part2_turns(
    bundle: DatasetBundle,
    setup: ModelSetup,
    csv_mode: str,
    max_csv_rows: int,
) -> tuple[str, list[PromptTurn]]:
    scenario_dir = SCENARIO_DIRS[setup.scenario]
    system_prompt = _fill_common((scenario_dir / "system_instruction.txt").read_text(encoding="utf-8"), bundle, setup)
    turns: list[PromptTurn] = []

    for turn_id, label, file_name in TURN_FILES:
        raw = (scenario_dir / file_name).read_text(encoding="utf-8")
        content = _fill_common(raw, bundle, setup)
        if turn_id == 0:
            content = _append_csv(content, bundle, csv_mode, max_csv_rows)
        if turn_id == 3:
            content = _append_script_execution_context(content)
        turns.append(PromptTurn(turn_id, label, content))

    return system_prompt, turns


def _fill_common(text: str, bundle: DatasetBundle, setup: ModelSetup) -> str:
    text = text.replace("<PRIMARY_TARGET_COLUMN_NAME>", bundle.target_col)
    text = text.replace("<PRIMARY_TARGET_COLUMN_NAME>", bundle.target_col)
    text = text.replace("<PRIMARY_TARGET_COLUMN_NAME>", bundle.target_col)
    text = text.replace("<MODEL_NAME>", setup.model_name)
    text = text.replace("<HYPERPARAMETERS>", setup.hyperparameters_text)
    text = text.replace("Forecast horizon: H (>0)", f"Forecast horizon: {setup.horizon_or_block} (>0)")
    text = text.replace("Block size: B (>0)", f"Block size: {setup.horizon_or_block} (>0)")
    text = text.replace("Forecast horizon: B (forecast", f"Forecast horizon: {setup.horizon_or_block} (forecast")
    return text


def _append_csv(content: str, bundle: DatasetBundle, csv_mode: str, max_csv_rows: int) -> str:
    csv_text = csv_for_prompt(bundle, mode=csv_mode, max_rows=max_csv_rows)
    return (
        f"{content}\n\n"
        "The raw CSV content is provided below and must be treated as the attached CSV file.\n"
        "RAW CSV START\n"
        f"{csv_text}"
        "RAW CSV END"
    )


def _append_script_execution_context(content: str) -> str:
    return (
        f"{content}\n\n"
        "For OPTION 2 only, read the dataset from a local CSV file named input.csv in the current working directory.\n"
        "Do not use absolute paths in generated code."
    )
