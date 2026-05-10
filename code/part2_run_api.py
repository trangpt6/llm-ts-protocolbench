from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from tqdm import tqdm

# Import from part2_api package
from part2_api import api_client, datasets, logging_utils, model_setup, paths, runner, settings

ClientManager = api_client.ClientManager
load_dataset = datasets.load_dataset
setup_logger = logging_utils.setup_logger
load_model_setup = model_setup.load_model_setup
normalise_scenario = model_setup.normalise_scenario

CHAT_LOG_DIR = paths.CHAT_LOG_DIR
DEFAULT_API_KEYS_PATH = paths.DEFAULT_API_KEYS_PATH
DEFAULT_MODEL_SETUP_PATH = paths.DEFAULT_MODEL_SETUP_PATH
SYSTEM_LOG_DIR = paths.SYSTEM_LOG_DIR

RunOptions = runner.RunOptions
run_part2_single = runner.run_part2_single

DATASET_SPECS = settings.DATASET_SPECS
DEFAULT_DELAY_BETWEEN_RUNS = settings.DEFAULT_DELAY_BETWEEN_RUNS
DEFAULT_DELAY_BETWEEN_TURNS = settings.DEFAULT_DELAY_BETWEEN_TURNS
DEFAULT_NUM_RUNS = settings.DEFAULT_NUM_RUNS
DEFAULT_PROVIDER_ORDER = settings.DEFAULT_PROVIDER_ORDER
MODEL_TRACKS = settings.MODEL_TRACKS
MODEL_TRACK_LABELS = settings.MODEL_TRACK_LABELS
SCENARIO_DIRS = settings.SCENARIO_DIRS


logger = setup_logger("run_part2_api", SYSTEM_LOG_DIR / "run-logs")


def main() -> None:
    args = parse_args()
    datasets = _select_datasets(args.dataset)
    scenarios = _select_scenarios(args.scenario)
    model_tracks = _select_model_tracks(args.model_track)
    num_runs = args.num_runs

    options = RunOptions(
        run_summary_path=args.system_log_dir / "run-summary-part2-interactive-llm-forecasting.csv",
        turn_metrics_path=args.system_log_dir / "turn-metrics-part2-interactive-llm-forecasting.csv",
        chat_log_dir=args.chat_log_dir,
        csv_mode=args.csv_mode,
        max_csv_rows=args.max_csv_rows,
        delay_between_turns=args.delay_between_turns,
        force=args.force,
    )

    if args.dry_run:
        providers = args.provider or DEFAULT_PROVIDER_ORDER
    else:
        manager = ClientManager(args.api_keys)
        providers = _select_providers(args.provider, manager.list_providers())

    total = len(datasets) * len(scenarios) * len(model_tracks) * len(providers) * num_runs
    logger.info(
        "Part 2 API plan: datasets=%d scenarios=%d tracks=%d providers=%d runs=%d total=%d",
        len(datasets),
        len(scenarios),
        len(model_tracks),
        len(providers),
        num_runs,
        total,
    )

    if args.dry_run:
        for dataset in datasets:
            for scenario in scenarios:
                for track in model_tracks:
                    setup = load_model_setup(args.model_setup, dataset, scenario, track)
                    for provider in providers:
                        for run_id in range(args.run_id_start, args.run_id_start + num_runs):
                            print(
                                f"Would run dataset={setup.dataset} scenario={setup.scenario} "
                                f"track={MODEL_TRACK_LABELS[track]} fixed_model={setup.model_name} provider={provider} run_id={run_id}"
                            )
        return

    tasks = [
        (dataset, scenario, track, provider, run_id)
        for dataset in datasets
        for scenario in scenarios
        for track in model_tracks
        for provider in providers
        for run_id in range(args.run_id_start, args.run_id_start + num_runs)
    ]
    setup_cache = {}
    dataset_cache = {}
    completed = 0
    failed = 0
    skipped = 0
    with tqdm(total=len(tasks), desc="Part2 API chats", unit="run") as progress:
        for dataset, scenario, track, provider, run_id in tasks:
            setup_key = (dataset, scenario, track)
            if setup_key not in setup_cache:
                setup_cache[setup_key] = load_model_setup(args.model_setup, dataset, scenario, track)
            setup = setup_cache[setup_key]

            dataset_key = (setup.dataset, setup.target_col, args.dataset_source)
            if dataset_key not in dataset_cache:
                dataset_cache[dataset_key] = load_dataset(setup.dataset, setup.target_col, source=args.dataset_source)
            bundle = dataset_cache[dataset_key]
            client = manager.get(provider)
            logger.info(
                "dataset=%s scenario=%s track=%s provider=%s run_id=%d",
                setup.dataset,
                setup.scenario,
                MODEL_TRACK_LABELS[track],
                provider,
                run_id,
            )
            result = run_part2_single(client, bundle, setup, run_id, options)
            if result.get("skipped"):
                skipped += 1
            elif result.get("status") == "CHAT_DONE":
                completed += 1
            else:
                failed += 1
            progress.update(1)
            if args.delay_between_runs:
                time.sleep(args.delay_between_runs)

    logger.info("Finished Part 2 API: completed=%d failed=%d skipped=%d", completed, failed, skipped)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Part 2 interactive forecasting through LLM APIs.")
    parser.add_argument("--dataset", action="append", help="Dataset name. Repeat or omit for all local project datasets.")
    parser.add_argument("--scenario", action="append", help="Scenario: S1, S2, S3, S4. Repeat or omit for all.")
    parser.add_argument("--provider", action="append", help="Provider from config/api_keys.json. Repeat or omit for all configured.")
    parser.add_argument("--model-track", action="append", choices=list(MODEL_TRACKS), help="Model setup track.")
    parser.add_argument("--num-runs", type=int, default=DEFAULT_NUM_RUNS)
    parser.add_argument("--run-id-start", type=int, default=1)
    parser.add_argument("--api-keys", type=Path, default=DEFAULT_API_KEYS_PATH)
    parser.add_argument("--model-setup", type=Path, default=DEFAULT_MODEL_SETUP_PATH)
    parser.add_argument("--dataset-source", choices=["auto", "local", "darts"], default="auto")
    parser.add_argument("--csv-mode", choices=["full", "head_tail", "metadata_only"], default="full")
    parser.add_argument("--max-csv-rows", type=int, default=400)
    parser.add_argument("--delay-between-turns", type=float, default=DEFAULT_DELAY_BETWEEN_TURNS)
    parser.add_argument("--delay-between-runs", type=float, default=DEFAULT_DELAY_BETWEEN_RUNS)
    parser.add_argument("--chat-log-dir", type=Path, default=CHAT_LOG_DIR)
    parser.add_argument("--run-summary-dir", type=Path, default=SYSTEM_LOG_DIR)
    parser.add_argument("--run-log-dir",    type=Path, default=SYSTEM_LOG_DIR / "run-logs")
    parser.add_argument("--force", action="store_true", help="Rerun even if checkpoint marks the run as done.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.num_runs <= 0:
        raise ValueError("--num-runs must be positive.")
    if args.run_id_start <= 0:
        raise ValueError("--run-id-start must be positive.")
    return args


def _select_datasets(values: list[str] | None) -> list[str]:
    valid = list(DATASET_SPECS)
    if not values:
        return valid
    selected = []
    for value in values:
        match = next((name for name in valid if name.lower() == value.lower()), None)
        if not match:
            raise ValueError(f"Unknown dataset '{value}'. Valid: {valid}")
        selected.append(match)
    return selected


def _select_scenarios(values: list[str] | None) -> list[str]:
    valid = list(SCENARIO_DIRS)
    if not values:
        return valid
    selected = [normalise_scenario(value) for value in values]
    invalid = [value for value in selected if value not in valid]
    if invalid:
        raise ValueError(f"Unknown scenario(s): {invalid}. Valid: {valid}")
    return selected


def _select_model_tracks(values: list[str] | None) -> list[str]:
    return values or list(MODEL_TRACKS)


def _select_providers(values: list[str] | None, configured: list[str]) -> list[str]:
    if values:
        missing = [value for value in values if value not in configured]
        if missing:
            raise ValueError(f"Provider(s) not configured: {missing}. Configured: {configured}")
        return values
    ordered = [provider for provider in DEFAULT_PROVIDER_ORDER if provider in configured]
    return ordered + [provider for provider in configured if provider not in ordered]


if __name__ == "__main__":
    logging.getLogger("httpx").setLevel(logging.WARNING)
    main()
