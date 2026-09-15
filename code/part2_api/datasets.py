from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import importlib

import numpy as np
import pandas as pd

from .settings import DATASET_SPECS, DatasetSpec


DARTS_DATASET_MAP = {
    "AirPassengers": "AirPassengersDataset",
    "ETTh1": "ETTh1Dataset",
    "ILINet": "ILINetDataset(multivariate=True)",
    "IceCreamHeater": "IceCreamHeaterDataset",
    "Temperature": "TemperatureDataset",
}


@dataclass
class DatasetBundle:
    name: str
    raw_df: pd.DataFrame
    target_col: str
    date_col: str | None
    source: str
    # Seasonal period (m) for the MASE scaling factor; see DatasetSpec.
    seasonal_period: int = 1

    @property
    def split_idx(self) -> int:
        return int(len(self.raw_df) * 0.8)

    @property
    def train_df(self) -> pd.DataFrame:
        return self.raw_df.iloc[: self.split_idx].copy()

    @property
    def test_df(self) -> pd.DataFrame:
        return self.raw_df.iloc[self.split_idx :].copy()

    def test_target_values(self) -> np.ndarray:
        return pd.to_numeric(self.test_df[self.target_col], errors="coerce").to_numpy(dtype=float)

    def train_target_values(self) -> np.ndarray:
        return pd.to_numeric(self.train_df[self.target_col], errors="coerce").to_numpy(dtype=float)

    def write_input_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_df.to_csv(path, index=False)


def load_dataset(dataset_name: str, target_col: str, source: str = "auto") -> DatasetBundle:
    spec = _get_spec(dataset_name)
    if source not in {"auto", "local", "darts"}:
        raise ValueError("dataset source must be one of: auto, local, darts")

    use_local = source in {"auto", "local"} and spec.local_path is not None and spec.local_path.exists()
    if use_local:
        raw_df = pd.read_csv(spec.local_path)
        dataset_source = f"local:{spec.local_path}"
        date_col = spec.date_col
    elif spec.darts_name:
        raw_df, date_col = _load_darts(spec.darts_name)
        dataset_source = f"darts:{spec.darts_name}"
    else:
        raise FileNotFoundError(f"No local file or Darts mapping found for dataset '{dataset_name}'.")

    """Special-case for ETTh1: keep only the date_col and the OT (target) column"""
    if dataset_name.lower() == "etth1" and date_col and date_col in raw_df.columns:
        raw_df = raw_df[[date_col, "OT"]]

    if raw_df.empty:
        raise ValueError(f"Dataset '{dataset_name}' is empty.")
    if target_col not in raw_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {dataset_name}. Columns: {list(raw_df.columns)}")
    if date_col and date_col not in raw_df.columns:
        raise ValueError(f"Date column '{date_col}' not found in {dataset_name}. Columns: {list(raw_df.columns)}")
    if int(len(raw_df) * 0.8) in {0, len(raw_df)}:
        raise ValueError(f"Dataset '{dataset_name}' is too short for an 80/20 split.")

    return DatasetBundle(
        name=dataset_name,
        raw_df=raw_df,
        target_col=target_col,
        date_col=date_col,
        source=dataset_source,
        seasonal_period=spec.seasonal_period,
    )


def csv_for_prompt(bundle: DatasetBundle, mode: str = "full", max_rows: int = 400) -> str:
    if mode == "full":
        return bundle.raw_df.to_csv(index=False)
    if mode == "head_tail":
        if len(bundle.raw_df) <= max_rows:
            return bundle.raw_df.to_csv(index=False)
        half = max(max_rows // 2, 1)
        head = bundle.raw_df.head(half)
        tail = bundle.raw_df.tail(half)
        omitted = pd.DataFrame([{col: "...ROWS_OMITTED..." for col in bundle.raw_df.columns}])
        return pd.concat([head, omitted, tail], ignore_index=True).to_csv(index=False)
    if mode == "metadata_only":
        return bundle.raw_df.head(min(max_rows, len(bundle.raw_df))).to_csv(index=False)
    raise ValueError("csv mode must be one of: full, head_tail, metadata_only")


def _get_spec(dataset_name: str) -> DatasetSpec:
    for name, spec in DATASET_SPECS.items():
        if name.lower() == dataset_name.lower():
            return spec
    raise KeyError(f"Unknown dataset '{dataset_name}'. Valid: {list(DATASET_SPECS)}")


def _load_darts(darts_name: str) -> tuple[pd.DataFrame, str]:
    class_name = DARTS_DATASET_MAP.get(darts_name.lower())
    if not class_name:
        raise KeyError(f"Darts dataset mapping not found: {darts_name}")

    darts_datasets = importlib.import_module("darts.datasets")
    if not hasattr(darts_datasets, class_name):
        raise ValueError(f"Current Darts install does not expose dataset class '{class_name}'.")

    series = getattr(darts_datasets, class_name)().load()
    df = series.pd_dataframe().reset_index()
    date_col = str(df.columns[0])
    return df, date_col

