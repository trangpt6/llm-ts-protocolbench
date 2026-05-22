# llm-ts-protocolbench

`llm-ts-protocolbench` is a research benchmark for evaluating time-series forecasting with LLMs, with an emphasis on protocol following, executable forecast generation, and comparisons against traditional baselines. The repository contains scripts for traditional baseline forecasting, LLM-derived model setup analysis, interactive Part 2 API experiments, chat-log parsing, forecast execution, metric computation, failure analysis, Part 0 comparisons, and visualization generation.

## Repository Structure

- `code/`: main pipeline scripts.
- `code/part2_api/`: support package for the Part 2 API runner, including API clients, dataset loading, prompt construction, checkpoints, logging, and single-run orchestration.
- `config/`: provider and API-key configuration. Use `api_keys.example.json` as the template for `api_keys.json`.
- `data/`: benchmark datasets: `AirPassengers`, `ETTh1`, `IceCreamHeater`, `ILINet`, and `Temperature`.
- `prompts/`: prompt templates for Part 1 and Part 2.
- `results/`: output artifacts, including model setup files, forecast outputs, metrics, summary tables, and visualizations.
- `logs/`: chat logs, master logs, system logs, and run logs.
- `docs/`: supporting research documentation.

## Environment

The project is intended for Python 3.11. Create and install the environment with:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

API keys are not required if you only run parsing, metric, summary, or visualization steps from existing artifacts. To run Part 2 through provider APIs, copy the example configuration and fill in real keys:

```powershell
Copy-Item .\config\api_keys.example.json .\config\api_keys.json
```

`config/api_keys.json` can reference environment variables with entries such as `"env:OPENAI_API_KEY"` or contain direct keys. The code supports OpenAI-compatible APIs, Anthropic, and Google Generative AI.

## Datasets and Scenarios

The benchmark uses five local datasets in `data/`:

- `AirPassengers`
- `ETTh1`
- `IceCreamHeater`
- `ILINet`
- `Temperature`

Part 2 uses four forecasting scenarios:

- `S1`: one-step static forecasting, train once, no test-set ground truth during forecasting.
- `S2`: one-step rolling forecasting, update at each step with ground truth.
- `S3`: multi-step rolling forecasting, with the horizon defined by the dataset/scenario setup.
- `S4`: block-wise rolling forecasting, retraining by block with ground truth.

Part 2 uses three model tracks:

- `baseline` -> `Base`
- `challenger_ml` -> `ChalML`
- `challenger_dl` -> `ChalDL`

The fixed Part 2 setup file is `results/part1-llm-strategic-consultation/part2-model-setup.csv`.

## Main Pipeline

### 1. Run Part 0 Traditional Baselines

```powershell
python .\code\part0_run_traditional_baselines.py
```

You can restrict datasets, scenarios, or models:

```powershell
python .\code\part0_run_traditional_baselines.py --datasets AirPassengers,Temperature --scenarios S1,S2 --models "Naive (Flat),Auto-ARIMA"
```

Outputs:

- `results/part0-traditional-baselines/baseline-traditional-results.csv`
- `results/part0-traditional-baselines/forecast-outputs/`

### 2. Test Part 2 API Providers

```powershell
python .\code\part2_0_test_apis.py --provider deepseek
```

Omit `--provider` to test all valid providers in `config/api_keys.json`.

### 3. Run the Part 2 API Runner

Dry-run a planned batch:

```powershell
python .\code\part2_1_run_api.py --dataset AirPassengers --scenario S1 --provider deepseek --num-runs 1 --dry-run
```

Run one small batch:

```powershell
python .\code\part2_1_run_api.py --dataset AirPassengers --scenario S1 --provider deepseek --model-track baseline --num-runs 1
```

If `--dataset`, `--scenario`, `--provider`, or `--model-track` are omitted, the runner uses all valid configured combinations. Each combination runs three times by default.

Main outputs:

- Chat logs: `logs/chat-logs/part2-interactive-llm-forecasting/`
- Run summary: `logs/system-logs/run-summary-part2-interactive-llm-forecasting.csv`
- Turn metrics: `logs/system-logs/turn-metrics-part2-interactive-llm-forecasting.csv`
- Checkpoints: `checkpoints/part2-interactive-llm-forecasting/`

### 4. Parse Chat Logs and Build Forecast Artifacts

```powershell
python .\code\part2_2_parse_master_logs.py
```

Outputs:

- `logs/master-logs/master-log-part2-interactive-llm-forecasting.csv`
- `results/part2-interactive-llm-forecasting/scripts-output/`
- `results/part2-interactive-llm-forecasting/raw-forecasts/`

### 5. Execute Forecast Scripts and Compute Metrics

```powershell
python .\code\part2_3_compute_metrics.py
```

This script reads the parsed master log, executes Python-script outputs when needed, parses list outputs, computes `mae`, `rmse`, `smape`, `mase`, and `r2`, and adds Diebold-Mariano comparisons for challenger tracks against `Base`.

Outputs:

- `results/part2-interactive-llm-forecasting/metrics-part2-interactive-llm-forecasting.csv`
- `results/part2-interactive-llm-forecasting/diebold-mariano-part2-interactive-llm-forecasting.csv`
- `results/part2-interactive-llm-forecasting/forecast-outputs/`
- `results/part2-interactive-llm-forecasting/execution-logs/`

### 6. Build Summary Tables and Failure Analyses

```powershell
python .\code\part2_4_summary_tables.py
python .\code\part2_4_execution_failure_summary.py
python .\code\part2_5_summarize_protocol_following.py
```

Outputs are written to:

- `results/part2-interactive-llm-forecasting/summary-tables/`
- `results/part2-interactive-llm-forecasting/summary/`

### 7. Compare Part 2 Against Part 0

```powershell
python .\code\part2_5_compare_with_part0.py
```

Outputs are written to `results/part2-interactive-llm-forecasting/summary-tables/`, including comparison tables, win/tie/loss summaries, and Diebold-Mariano tests for Part 2 vs Part 0.

### 8. Generate Visualizations

```powershell
python .\code\part2_6_generate_visuals.py --phase all
```

Use `--phase pre`, `--phase post`, or `--dry-run` as needed.

Outputs:

- `results/visualizations/pre-results/`
- `results/visualizations/post-results/`

## Reproducible Execution Order

```powershell
python .\code\part0_run_traditional_baselines.py
python .\code\part2_0_test_apis.py --provider deepseek
python .\code\part2_1_run_api.py --provider deepseek --num-runs 1
python .\code\part2_2_parse_master_logs.py
python .\code\part2_3_compute_metrics.py
python .\code\part2_4_summary_tables.py
python .\code\part2_4_execution_failure_summary.py
python .\code\part2_5_summarize_protocol_following.py
python .\code\part2_5_compare_with_part0.py
python .\code\part2_6_generate_visuals.py --phase all
```

## Notes

- Do not commit `config/api_keys.json`.
- `results/` and `logs/` can become large; commit only the artifacts needed for the paper or thesis.
- LLM-generated forecast scripts may import `statsmodels`, `scikit-learn`, `xgboost`, `lightgbm`, or `torch`, so `requirements.txt` includes dependencies for both the main pipeline and forecast-script execution.
