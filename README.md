# llm-ts-protocolbench

`llm-ts-protocolbench` la benchmark nghien cuu ve kha nang tuan thu giao thuc va du bao chuoi thoi gian cua LLM. Repo gom pipeline chay baseline truyen thong, tong hop goi y thiet lap mo hinh tu LLM, chay thuc nghiem Part 2 qua API LLM, parse ket qua, thuc thi forecast script, tinh metric, so sanh voi baseline va tao bang/visualization cho bao cao.

## Cau truc repo

- `code/`: cac script chinh cua pipeline.
- `code/part2_api/`: package ho tro Part 2 API runner, gom client API, nap dataset, build prompt, checkpoint/logging va runner tung run.
- `config/`: cau hinh provider/API key. Dung `api_keys.example.json` lam mau cho `api_keys.json`.
- `data/`: cac dataset benchmark: `AirPassengers`, `ETTh1`, `IceCreamHeater`, `ILINet`, `Temperature`.
- `prompts/`: prompt template cho Part 1 va Part 2.
- `results/`: artifact dau ra, gom model setup, forecast output, metrics, summary tables va visualization.
- `logs/`: chat logs, master logs va system/run logs.
- `docs/`: tai lieu phu tro cho nghien cuu.

## Moi truong

Repo duoc viet cho Python 3.11. Tao va cai moi truong:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Neu chi chay cac buoc parse/tong hop da co san artifact thi khong can API key. Neu chay Part 2 qua API, sao chep file mau va dien key:

```powershell
Copy-Item .\config\api_keys.example.json .\config\api_keys.json
```

`config/api_keys.json` co the dung bien moi truong theo dang `"env:OPENAI_API_KEY"` hoac dien truc tiep key. Cac provider dang duoc ho tro trong code gom OpenAI-compatible APIs, Anthropic va Google Generative AI.

## Dataset va kich ban

Benchmark chay tren 5 dataset local trong `data/`:

- `AirPassengers`
- `ETTh1`
- `IceCreamHeater`
- `ILINet`
- `Temperature`

Bon kich ban Part 2:

- `S1`: one-step static, train mot lan, khong dung ground truth test khi du bao.
- `S2`: one-step rolling, cap nhat theo tung buoc voi ground truth.
- `S3`: multi-step rolling, horizon theo setup cua dataset/scenario.
- `S4`: block-wise rolling, retrain theo block voi ground truth.

Ba nhanh mo hinh Part 2:

- `baseline` -> `Base`
- `challenger_ml` -> `ChalML`
- `challenger_dl` -> `ChalDL`

Setup co dinh cho Part 2 nam tai `results/part1-llm-strategic-consultation/part2-model-setup.csv`.

## Pipeline chinh

### 1. Chay Part 0 traditional baselines

```powershell
python .\code\part0_run_traditional_baselines.py
```

Co the loc dataset/scenario/model:

```powershell
python .\code\part0_run_traditional_baselines.py --datasets AirPassengers,Temperature --scenarios S1,S2 --models "Naive (Flat),Auto-ARIMA"
```

Dau ra:

- `results/part0-traditional-baselines/baseline-traditional-results.csv`
- `results/part0-traditional-baselines/forecast-outputs/`

### 2. Test API provider cho Part 2

```powershell
python .\code\part2_0_test_apis.py --provider deepseek
```

Bo `--provider` neu muon test tat ca provider hop le trong `config/api_keys.json`.

### 3. Chay Part 2 API runner

Dry-run de xem ke hoach:

```powershell
python .\code\part2_1_run_api.py --dataset AirPassengers --scenario S1 --provider deepseek --num-runs 1 --dry-run
```

Chay that mot batch nho:

```powershell
python .\code\part2_1_run_api.py --dataset AirPassengers --scenario S1 --provider deepseek --model-track baseline --num-runs 1
```

Neu bo cac tham so `--dataset`, `--scenario`, `--provider`, `--model-track`, runner se chay tat ca combination hop le theo cau hinh. Mac dinh moi combination chay 3 lan.

Dau ra chinh:

- Chat logs: `logs/chat-logs/part2-interactive-llm-forecasting/`
- Run summary: `logs/system-logs/run-summary-part2-interactive-llm-forecasting.csv`
- Turn metrics: `logs/system-logs/turn-metrics-part2-interactive-llm-forecasting.csv`
- Checkpoints: `checkpoints/part2-interactive-llm-forecasting/`

### 4. Parse chat logs va tao artifact forecast

```powershell
python .\code\part2_2_parse_master_logs.py
```

Dau ra:

- `logs/master-logs/master-log-part2-interactive-llm-forecasting.csv`
- `results/part2-interactive-llm-forecasting/scripts-output/`
- `results/part2-interactive-llm-forecasting/raw-forecasts/`

### 5. Thuc thi forecast script va tinh metrics

```powershell
python .\code\part2_3_compute_metrics.py
```

Script nay doc master log, thuc thi output dang Python script khi can, parse output dang list, tinh `mae`, `rmse`, `smape`, `mase`, `r2`, va tinh Diebold-Mariano cho challenger vs Base.

Dau ra:

- `results/part2-interactive-llm-forecasting/metrics-part2-interactive-llm-forecasting.csv`
- `results/part2-interactive-llm-forecasting/diebold-mariano-part2-interactive-llm-forecasting.csv`
- `results/part2-interactive-llm-forecasting/forecast-outputs/`
- `results/part2-interactive-llm-forecasting/execution-logs/`

### 6. Tao summary tables va phan tich loi

```powershell
python .\code\part2_4_summary_tables.py
python .\code\part2_4_execution_failure_summary.py
python .\code\part2_5_summarize_protocol_following.py
```

Dau ra nam trong:

- `results/part2-interactive-llm-forecasting/summary-tables/`
- `results/part2-interactive-llm-forecasting/summary/`

### 7. So sanh Part 2 voi Part 0

```powershell
python .\code\part2_5_compare_with_part0.py
```

Dau ra duoc ghi vao `results/part2-interactive-llm-forecasting/summary-tables/`, gom bang comparison, win/tie/loss va Diebold-Mariano Part 2 vs Part 0.

### 8. Tao visualization

```powershell
python .\code\part2_6_generate_visuals.py --phase all
```

Co the dung `--phase pre`, `--phase post`, hoac `--dry-run`.

Dau ra:

- `results/visualizations/pre-results/`
- `results/visualizations/post-results/`

## Thu tu chay de tai tao ket qua

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

## Ghi chu

- `config/api_keys.json` khong nen commit len GitHub.
- `results/` va `logs/` co the rat lon; can can nhac chi commit artifact can thiet cho paper/thesis.
- Cac forecast script do LLM sinh ra co the import `statsmodels`, `scikit-learn`, `xgboost`, `lightgbm` hoac `torch`, nen `requirements.txt` bao gom ca dependency cho moi truong thuc thi script.
