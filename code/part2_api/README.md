# Part 2 API Runner

Thu muc nay la package ho tro `code/part2_1_run_api.py`, dung de chay Part 2 Interactive LLM Forecasting qua API provider.

## Thanh phan

- `api_client.py`: nap cau hinh provider, xoay API key, retry khi gap rate limit/timeout/API error, goi OpenAI-compatible, Anthropic va Google API.
- `datasets.py`: nap dataset tu `data/` hoac tu `darts.datasets`, tao train/test split va CSV payload cho prompt.
- `model_setup.py`: doc `results/part1-llm-strategic-consultation/part2-model-setup.csv` va lay fixed model/hyperparameters theo dataset, scenario, track.
- `prompts.py`: ghep prompt template Part 2 cho 4 turn.
- `runner.py`: chay mot run gom Turn 0 den Turn 3, validate output, ghi chat log, run summary, turn metrics va checkpoint.
- `settings.py`: cau hinh dataset, scenario, model track, timeout, max token va status taxonomy.
- `paths.py`: cac duong dan chuan trong repo.
- `logging_utils.py`: tao logger file/console.

## Cau hinh API key

Tao file config tu mau:

```powershell
Copy-Item .\config\api_keys.example.json .\config\api_keys.json
```

Moi provider co dang:

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

`provider_type` hop le:

- `openai_compatible`
- `anthropic`
- `google`

`keys` co the la key truc tiep hoac `env:VARIABLE_NAME`. Neu mot bien moi truong chua nhieu key, co the ngan cach bang dau phay.

## Quick Start

Test provider:

```powershell
python .\code\part2_0_test_apis.py --provider deepseek
```

Dry-run mot batch:

```powershell
python .\code\part2_1_run_api.py --dataset AirPassengers --scenario S1 --provider deepseek --num-runs 1 --dry-run
```

Chay mot combination:

```powershell
python .\code\part2_1_run_api.py --dataset AirPassengers --scenario S1 --provider deepseek --model-track baseline --num-runs 1
```

Chay toan bo provider/dataset/scenario/track da cau hinh:

```powershell
python .\code\part2_1_run_api.py
```

## Tuy chon quan trong

- `--dataset`: lap lai de chon nhieu dataset; bo qua de chay tat ca dataset local.
- `--scenario`: `S1`, `S2`, `S3`, `S4`; bo qua de chay tat ca.
- `--provider`: ten provider trong `config/api_keys.json`; bo qua de chay tat ca provider hop le.
- `--model-track`: `baseline`, `challenger_ml`, `challenger_dl`; bo qua de chay ca 3 track.
- `--num-runs`: so lan lap moi combination, mac dinh 3.
- `--run-id-start`: run id bat dau, mac dinh 1.
- `--dataset-source`: `auto`, `local`, hoac `darts`.
- `--csv-mode`: `full`, `head_tail`, hoac `metadata_only`.
- `--max-csv-rows`: gioi han so dong khi `csv-mode=head_tail`.
- `--force`: chay lai ca run da co checkpoint thanh cong.
- `--dry-run`: chi in ke hoach, khong goi API.

## Input

- Prompt templates: `prompts/part2-interactive-llm-forecasting/`
- Dataset CSV: `data/`
- Model setup: `results/part1-llm-strategic-consultation/part2-model-setup.csv`
- API config: `config/api_keys.json`

## Output

- Chat logs: `logs/chat-logs/part2-interactive-llm-forecasting/Part2-<Dataset>-<Scenario>-<Branch>-<LLMVersion>-Run<N>.txt`
- Run summary: `logs/system-logs/run-summary-part2-interactive-llm-forecasting.csv`
- Turn metrics: `logs/system-logs/turn-metrics-part2-interactive-llm-forecasting.csv`
- Run logs: `logs/system-logs/run-logs/`
- Checkpoints: `checkpoints/part2-interactive-llm-forecasting/`

Sau khi thu thap chat logs, tiep tuc pipeline:

```powershell
python .\code\part2_2_parse_master_logs.py
python .\code\part2_3_compute_metrics.py
python .\code\part2_4_summary_tables.py
python .\code\part2_4_execution_failure_summary.py
python .\code\part2_5_summarize_protocol_following.py
python .\code\part2_5_compare_with_part0.py
```
