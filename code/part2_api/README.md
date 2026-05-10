# Part 2 API Runner

This folder contains the local API runner for `Part 2 - Interactive Forecasting`.

## Quick Start

1. Put API keys in `config/api_keys.json`.

Use environment variables:

```json
"keys": ["env:DEEPSEEK_API_KEY", "env:DEEPSEEK_API_KEY_2"]
```

Or direct keys:

```json
"keys": ["sk-..."]
```

2. Test API connectivity.

```powershell
.\.venv\Scripts\python.exe .\code\test_part2_apis.py --provider deepseek
```

3. Dry-run the experiment plan.

```powershell
.\.venv\Scripts\python.exe .\code\run_part2_api.py --dataset AirPassengers --scenario S1 --provider deepseek --num-runs 1 --dry-run
```

4. Run one Part 2 experiment.

```powershell
.\.venv\Scripts\python.exe .\code\run_part2_api.py --dataset AirPassengers --scenario S1 --provider deepseek --num-runs 1
```

## Pipeline

The API runner only calls the LLM through Turn 3, then immediately writes the chat file and stops API interaction for that run.

After all chats are collected:

```powershell
.\.venv\Scripts\python.exe .\code\part2_parse_master_logs.py
.\.venv\Scripts\python.exe .\code\part2_compute_metrics.py
.\.venv\Scripts\python.exe .\code\part2_summary_tables.py
```

## Outputs

- Chat logs: `logs/chat-logs/part2-interactive-llm-forecasting/Part2_<Dataset>_<Scenario>_<Base|ChalML|ChalDL>_<LLMName>_Run<N>.txt`
- API run master log: `logs/master-logs/master-log-part2-interactive-llm-forecasting.csv`
- API turn-level log: `logs/master-logs/turn-log-part2-interactive-llm-forecasting.csv`
- Parsed chat master log: `logs/master-logs/master-log-part2-parsed.csv`
- Scripts/lists parsed from Turn 3: `results/part2-interactive-llm-forecasting/scripts-output/` and `results/part2-interactive-llm-forecasting/raw-forecasts/`
- Metrics: `results/part2-interactive-llm-forecasting/part2-metrics.csv`
- Summary tables: `results/part2-interactive-llm-forecasting/summary-tables/`

The runner reads prompt templates from `prompts/part2-interactive-llm-forecasting/` and fixed model setups from `results/part1-llm-strategic-consultation/part2-model-setup.csv`.
By default it runs all 3 forecast model tracks (`Base`, `ChalML`, `ChalDL`) and 3 repeated runs per combination.
