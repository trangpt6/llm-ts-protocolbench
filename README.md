# Time Series Research Project (Template)

This template sets up a clean structure for your time series & forecasting research.

## Structure

- `notebooks/`: main analysis notebooks
- `data/`: raw/processed datasets
- `results/`: saved figures, metrics
- `scripts/`: optional python scripts
- `.venv/`: your isolated Python environment (not included here)
- `run_lab.ps1`: quick-launch script (activate .venv and open VS Code)

## How to use

1. Place this folder where you want (e.g., `C:\Users\Trang\TimeSeriesResearch`).
2. Create/activate a virtual environment inside this folder:
   ```powershell
   py -3.11 -m venv .venv
   .\.venv\Scripts\activate
   ```
3. Open the project:
   - Either double-click `run_lab.ps1`
   - Or `code .` from this folder
4. Start with `notebooks/00_intro_setup.ipynb` and run all cells.

## Part 2 API Runner

Part 2 interactive forecasting can be run locally through provider APIs.

```powershell
.\.venv\Scripts\python.exe .\code\test_part2_apis.py --provider deepseek
.\.venv\Scripts\python.exe .\code\run_part2_api.py --dataset AirPassengers --scenario S1 --provider deepseek --num-runs 1 --dry-run
.\.venv\Scripts\python.exe .\code\run_part2_api.py --dataset AirPassengers --scenario S1 --provider deepseek --num-runs 1
```

API keys live in `config/api_keys.json`. Prompt templates are read from `prompts/Part2_Interactive_LLM_Forecasting/`, fixed model setups from `results/Part1_LLM_Strategic_Consultation/part2_model_setup.csv`, and outputs are written into the existing Part 2 folders.
Chat files are written to `logs/chats/Part2_Interactive_LLM_Forecasting/` using names like `Part2_ILINet_S1_Base_ClaudeSonnet46_Run1.txt`. After chats finish, run `code/part2_parse_master_logs.py`, `code/part2_compute_metrics.py`, then `code/part2_summary_tables.py`.
