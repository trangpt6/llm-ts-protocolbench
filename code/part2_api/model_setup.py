from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .settings import MODEL_TRACKS


@dataclass(frozen=True)
class ModelSetup:
    dataset: str
    scenario: str
    target_col: str
    horizon_or_block: int
    scenario_description: str
    model_track: str
    model_name: str
    hyperparameters_text: str
    hyperparameters: dict[str, Any]


def load_model_setup(path: Path, dataset: str, scenario: str, model_track: str) -> ModelSetup:
    scenario = normalise_scenario(scenario)
    if model_track not in MODEL_TRACKS:
        raise KeyError(f"Unknown model_track '{model_track}'. Valid: {list(MODEL_TRACKS)}")
    model_col, hp_col = MODEL_TRACKS[model_track]

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row["dataset"].lower() == dataset.lower() and normalise_scenario(row["scenario"]) == scenario:
                hp_text = row[hp_col].strip()
                hp = json.loads(hp_text)
                return ModelSetup(
                    dataset=row["dataset"],
                    scenario=scenario,
                    target_col=row["dataset_primary_target"].strip(),
                    horizon_or_block=int(float(row["horizon_or_blocksize"])),
                    scenario_description=row.get("scenario_description", "").strip(),
                    model_track=model_track,
                    model_name=row[model_col].strip(),
                    hyperparameters_text=repr(hp),
                    hyperparameters=hp,
                )

    raise LookupError(f"No model setup found for dataset={dataset}, scenario={scenario}, track={model_track}")


def load_all_model_setups(path: Path, dataset: str, scenario: str) -> list[ModelSetup]:
    return [load_model_setup(path, dataset, scenario, track) for track in MODEL_TRACKS]


def normalise_scenario(value: str) -> str:
    value = value.strip().upper()
    aliases = {
        "1": "S1",
        "2": "S2",
        "3": "S3",
        "4": "S4",
        "SCENARIO1": "S1",
        "SCENARIO2": "S2",
        "SCENARIO3": "S3",
        "SCENARIO4": "S4",
    }
    return aliases.get(value, value)
