# Part 2 API Runner

This package supports `code/part2_1_run_api.py`, which runs Part 2 Interactive LLM Forecasting through provider APIs.

## Components

- `api_client.py`: loads provider configuration, rotates API keys, retries rate-limit/timeout/API failures, and dispatches calls to OpenAI-compatible, Anthropic, and Google APIs.
- `datasets.py`: loads datasets from `data/` or `darts.datasets`, creates train/test splits, and builds CSV payloads for prompts.
- `model_setup.py`: reads `results/part1-llm-strategic-consultation/part2-model-setup.csv` and selects fixed models/hyperparameters by dataset, scenario, and track.
- `prompts.py`: builds the four-turn Part 2 prompt sequence from templates.
- `runner.py`: executes one run from Turn 0 through Turn 3, validates output, and writes chat logs, run summaries, turn metrics, and checkpoints.
- `settings.py`: stores dataset, scenario, model-track, timeout, max-token, and status-taxonomy configuration.
- `paths.py`: defines canonical repository paths.
- `logging_utils.py`: creates file and console loggers.

## API-Key Configuration

Create a config file from the example:

```powershell
Copy-Item .\config\api_keys.example.json .\config\api_keys.json
```

Each provider has this shape:

```json
{
  "enabled": true,
  "provider_type": "openai_compatible",
  "model_id": "deepseek-chat",
  "display_name": "DeepSeekV4",
  "base_url": "https://api.deepseek.com",
  "max_tokens": 4096,
  "rotate_after_success": true,
  "keys": ["env:DEEPSEEK_API_KEY"]
}
```

Valid `provider_type` values:

- `openai_compatible`
- `anthropic`
- `google`

`keys` can contain direct API keys or `env:VARIABLE_NAME` references. If an environment variable contains multiple keys, separate them with commas.

## Quick Start

Test one provider:

```powershell
python .\code\part2_0_test_apis.py --provider deepseek
```

Dry-run one batch:

```powershell
python .\code\part2_1_run_api.py --dataset AirPassengers --scenario S1 --provider deepseek --num-runs 1 --dry-run
```

Run one combination:

```powershell
python .\code\part2_1_run_api.py --dataset AirPassengers --scenario S1 --provider deepseek --model-track baseline --num-runs 1
```

Run all configured provider/dataset/scenario/track combinations:

```powershell
python .\code\part2_1_run_api.py
```

## Important Options

- `--dataset`: repeat to select multiple datasets; omit to run all local datasets.
- `--scenario`: `S1`, `S2`, `S3`, or `S4`; omit to run all scenarios.
- `--provider`: provider name from `config/api_keys.json`; omit to run all valid providers.
- `--model-track`: `baseline`, `challenger_ml`, or `challenger_dl`; omit to run all three tracks.
- `--num-runs`: repeated runs per combination; default is 3.
- `--run-id-start`: starting run id; default is 1.
- `--dataset-source`: `auto`, `local`, or `darts`.
- `--csv-mode`: `full`, `head_tail`, or `metadata_only`.
- `--max-csv-rows`: row limit when `csv-mode=head_tail`.
- `--force`: rerun combinations even if a successful checkpoint already exists.
- `--dry-run`: print the plan without calling any API.

## Inputs

- Prompt templates: `prompts/part2-interactive-llm-forecasting/`
- Dataset CSV files: `data/`
- Model setup: `results/part1-llm-strategic-consultation/part2-model-setup.csv`
- API config: `config/api_keys.json`

## Outputs

- Chat logs: `logs/chat-logs/part2-interactive-llm-forecasting/Part2-<Dataset>-<Scenario>-<Branch>-<LLMVersion>-Run<N>.txt`
- Run summary: `logs/system-logs/run-summary-part2-interactive-llm-forecasting.csv`
- Turn metrics: `logs/system-logs/turn-metrics-part2-interactive-llm-forecasting.csv`
- Run logs: `logs/system-logs/run-logs/`
- Checkpoints: `checkpoints/part2-interactive-llm-forecasting/`

After chat logs are collected, continue with:

```powershell
python .\code\part2_2_parse_master_logs.py
python .\code\part2_3_compute_metrics.py
python .\code\part2_4_summary_tables.py
python .\code\part2_4_execution_failure_summary.py
python .\code\part2_5_summarize_protocol_following.py
python .\code\part2_5_compare_with_part0.py
```
