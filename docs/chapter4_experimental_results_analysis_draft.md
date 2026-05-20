# Chapter 4. Experimental Results and Analysis

This chapter reports the empirical results of the proposed LLM-based time-series forecasting protocol. The analysis covers three levels of evidence: traditional forecasting baselines, zero-shot model recommendation behavior, and executable interactive forecasting performance. Unless stated otherwise, lower values of MAE, RMSE, sMAPE, and MASE indicate better forecasting accuracy, while higher R2 indicates better explanatory performance. The main Part 2 metric table contains 924 evaluated rows, of which 517 completed successfully and 407 failed during output parsing, code execution, or forecast validation.

Suggested figures and tables for this chapter:

- Figure 4.1: `results/visualizations/post-results/coverage_failure.png`
- Figure 4.2: `results/visualizations/post-results/scenario_metric_trend.png`
- Figure 4.3: `results/visualizations/post-results/part2_vs_part0_delta_rmse.png`
- Figure 4.4: `results/visualizations/post-results/dm_test_scatter.png`
- Table 4.1: `results/part0-traditional-baselines/baseline-traditional-results.csv`
- Table 4.2: `results/part1-llm-strategic-consultation/part2-model-setup.csv`
- Table 4.3: `results/visualizations/post-results/run_coverage.csv`
- Table 4.4: `results/visualizations/post-results/summary_by_dataset_scenario.csv`
- Table 4.5: `results/visualizations/post-results/summary_by_llm.csv`
- Table 4.6: `results/visualizations/post-results/paired_summary_vs_base_by_llm.csv`
- Table 4.7: `results/visualizations/post-results/part2_vs_part0_comparison.csv`

## 4.1 Results on traditional baselines

The traditional baseline stage provides a fixed reference point for interpreting the later LLM-generated forecasts. Three families of baselines were evaluated: flat naive forecasts, seasonal naive forecasts, and Auto-ARIMA where applicable. The best baseline varied substantially by dataset and scenario, confirming that no single classical reference model dominated across all conditions.

Across the 20 dataset-scenario combinations, Auto-ARIMA achieved the lowest RMSE in 6 cases, the flat naive model in 7 cases, and the seasonal naive model in 7 cases. Auto-ARIMA was strongest on some strongly seasonal or trend-sensitive settings, such as AirPassengers S1 and S4, IceCreamHeater S1 and S4, ILINet S4, and Temperature S4. However, simple naive baselines remained competitive or superior in rolling-update settings. For example, ETTh1 S2 and S3 were best served by the flat naive baseline with RMSE 0.654 and R2 0.964, while Temperature S2 and S3 were also best under the flat naive baseline with RMSE 2.481 and R2 0.635.

The baseline results show two important patterns. First, rolling one-step or short-horizon setups often reward very simple updating rules. Second, longer seasonal horizons do not automatically favor more complex models; seasonal naive forecasts were strong for AirPassengers S2-S3 and IceCreamHeater S2-S3. These results are important because they set a high practical bar for LLM-generated code: a generated solution must not only be syntactically correct, but must also outperform simple and robust baselines.

## 4.2 Zero-shot recommendation results

The zero-shot recommendation stage was used to lock in the model families and hyperparameters for the subsequent interactive forecasting experiment. For each dataset-scenario pair, 30 recommendation runs were aggregated to select a statistical baseline, a machine-learning challenger, and a deep-learning challenger.

The recommendation behavior was strongly biased toward statistical models. For most dataset-scenario pairs, the selected baseline was SARIMA, especially on ETTh1, ILINet, IceCreamHeater, and Temperature. AirPassengers was the main exception: Exponential Smoothing was selected for S1, S2, and S4, while SARIMA was selected for S3. Model-level consensus was often high. For ILINet, SARIMA received 83.3%-93.3% of votes across S1-S4. ETTh1 also showed high SARIMA support, ranging from 60.0% in S4 to 80.0% in S2. Temperature was more ambiguous, with SARIMA model votes falling from 70.0% in S2 to 43.3% in S4.

The locked-in challenger models followed a consistent pattern. Machine-learning challengers were usually LightGBM, XGBoost, or RandomForest, while deep-learning challengers were LSTM, GRU, or TCN. These choices are reasonable at the model-family level, but several hyperparameter choices later became problematic. The most important example is Temperature, where the recommended SARIMA baseline used a yearly seasonal period of 365. This was statistically plausible for daily temperature data but computationally expensive in the generated-code setting, causing many timeouts in Part 2.

Overall, the zero-shot recommendation stage produced plausible model selections, but the hyperparameter consensus was weaker than model-family consensus. For example, AirPassengers S3 had 50.0% model consensus for SARIMA but only 13.3% exact hyperparameter consensus. Temperature S3 and S4 had only 16.7% exact hyperparameter consensus. This gap indicates that LLMs can often identify broad model families, but are less stable when specifying executable hyperparameters.

## 4.3 Interactive forecasting results

The interactive forecasting stage evaluated whether LLMs could transform the locked-in setups into executable forecasting code or valid forecast lists. Out of 924 total Part 2 rows, 517 completed successfully, corresponding to an overall success rate of 55.95%. However, this aggregate masks large differences across datasets, scenarios, branches, and LLM models.

At the branch level, ChalML was the most robust branch, with 212 successful runs out of 300 or 70.67%. ChalDL achieved 172 successful runs out of 320 or 53.75%. Base had the weakest execution coverage, with 133 successful runs out of 304 or 43.75%. This result is notable because the Base branch often used statistical models selected in Part 1. In particular, computationally heavy SARIMA configurations made Base less executable than the machine-learning challenger branch.

### 4.3.1 Results by dataset

Dataset complexity was the strongest determinant of execution success. AirPassengers and IceCreamHeater were the most reliable datasets. AirPassengers achieved 189 successful runs out of 216, corresponding to 87.50% success. IceCreamHeater achieved 180 successful runs out of 216, corresponding to 83.33% success. These datasets are shorter and have clearer monthly seasonal structures, making them easier for generated scripts to handle.

ILINet and Temperature were much harder. ILINet achieved only 94 successful runs out of 216, or 43.52%. Temperature achieved 46 successful runs out of 216, or 21.30%. ETTh1 was the lowest-coverage dataset, with 8 successful runs out of 60, or 13.33%. ETTh1 combines multivariate structure, long test length, and scenario-specific update rules, which exposed weaknesses in generated code and computational feasibility.

Accuracy also varied strongly across datasets. On AirPassengers, the best LLM varied by scenario: ClaudeOpus47 was best for S1, S2, and S4, while DeepSeekV4Pro was best for S3. The best AirPassengers RMSE values were 77.633 in S1, 33.016 in S2, 29.816 in S3, and 62.571 in S4. On IceCreamHeater, DeepSeekV4Pro was best in S1 and S4, while ClaudeOpus47 was best in S2 and S3. The strongest IceCreamHeater result was S3 under ClaudeOpus47, with MAE 6.825 and RMSE 8.827.

For ILINet, the best results were concentrated in rolling-update settings. Gemini31Pro achieved the best S2 result with MAE 0.195, RMSE 0.337, and R2 0.951. GPT55 achieved the best RMSE and R2 for S3, with RMSE 0.374 and R2 0.940. Temperature produced some numerically strong challenger results despite low coverage: ClaudeOpus47 was best in S2 and S4, while Grok43 was best in S3 and KimiK26 in S1. However, the lack of successful Temperature Base runs limits the strength of challenger-vs-base conclusions for this dataset.

### 4.3.2 Results by scenario

Execution success differed by scenario. S1 had 151 successful runs out of 234, corresponding to 64.53%. S2 had 134 successful runs out of 230, or 58.26%. S3 was the weakest scenario, with only 93 successful runs out of 230, or 40.43%. S4 recovered to 139 successful runs out of 230, or 60.43%.

These results suggest that multi-step rolling forecasting with continuous retraining is the most difficult setting for LLM-generated implementations. S3 combines multi-step forecasting with rolling updates and ground-truth usage, requiring careful loop structure and correct output length. Many failures in this setting were caused by forecast-length mismatches or incorrect handling of the forecast horizon. By contrast, S1 and S2 are one-step scenarios and therefore easier to implement, although S1 may still suffer from recursive forecast logic when ground truth is disabled.

In accuracy terms, short-horizon rolling scenarios often performed better than static or long-horizon scenarios. ILINet S2, ILINet S3, Temperature S2, and Temperature S3 achieved relatively strong R2 values when successful. AirPassengers S3 also performed well under DeepSeekV4Pro, with R2 0.841. However, long block-wise scenarios such as AirPassengers S4 and ILINet S4 were less stable and more sensitive to accumulated errors.

### 4.3.3 Results by LLM model

Among the evaluated LLMs, ClaudeOpus47 achieved the highest execution success rate and the strongest aggregate metric profile. It completed 101 successful runs out of 148, or 68.24%, and had the lowest aggregate mean MAE (20.342), RMSE (25.753), sMAPE (21.760), and MASE (2.913). GPT55 was second in execution success with 98 successful runs out of 148, or 66.22%, and also had competitive aggregate metrics.

Gemini31Pro achieved 95 successful runs out of 149, or 63.76%. Grok43 and KimiK26 were near 50% success, with 49.67% and 49.32% respectively. DeepSeekV4Pro had the largest number of total rows because of available runs, but only 75 successful runs out of 180, or 41.67%. Its aggregate accuracy was mixed: it achieved some best-in-scenario results, especially AirPassengers S3, ETTh1 S2/S4, and IceCreamHeater S1/S4, but its lower execution robustness reduced its overall reliability.

The best LLM by dataset-scenario was not uniform. ClaudeOpus47 dominated several simpler or more stable settings, including AirPassengers S1/S2/S4, ILINet S4, IceCreamHeater S2/S3, and Temperature S2/S4. DeepSeekV4Pro produced several strong best-case results, especially AirPassengers S3, ETTh1 S2/S4, ILINet S3 by MAE, and IceCreamHeater S1/S4. Gemini31Pro was strongest on ETTh1 S1 and ILINet S2. This suggests that the most reliable model overall is not always the best model for every individual time-series setting.

### 4.3.4 Comparison with locked-in baselines

The paired challenger-vs-base analysis shows that the locked-in challenger branches did not consistently outperform the Base branch. For ChalML, 120 paired comparisons were available. The mean MAE increased from 13.549 in Base to 30.123 in ChalML, and the mean RMSE increased from 16.564 to 38.975. The mean R2 decreased from 0.559 to 0.140. Therefore, although ChalML had the highest execution success rate, it did not improve average accuracy over the Base branch.

The ChalDL branch performed substantially worse. Across 117 paired comparisons, mean MAE increased from 12.616 in Base to 143.750 in ChalDL, and mean RMSE increased from 15.386 to 152.837. Mean R2 dropped from 0.481 to -23.848. This indicates that generated deep-learning scripts were much less reliable as forecasting implementations under the current protocol. Common causes include unstable training loops, inappropriate scaling, shape errors, and poor horizon handling.

The Diebold-Mariano tests support this conclusion. In the internal challenger-vs-base DM results, only 3 challenger runs were significantly better than Base, while 234 were not significant or worse. All 3 significant wins came from ChalDL, but this was not enough to offset the broad degradation in average performance. Thus, the locked-in statistical Base remained the most reliable forecasting branch when executable.

## 4.4 Protocol compliance analysis

Protocol compliance was assessed across the four turns of the interactive workflow. The results show strong compliance in early descriptive turns but weaker compliance in scenario reasoning and executable code generation.

### 4.4.1 Turn-wise compliance rates

Turn 0 and Turn 1 were highly reliable. Across all LLMs, Turn 0 format-following was effectively perfect, with 100% scores for all models. Turn 1 remained strong for most models: ClaudeOpus47 scored 99.32, GPT55 100.00, KimiK26 96.85, and Gemini31Pro 100.00. Grok43 and DeepSeekV4Pro were weaker on Turn 1, both scoring 72.41.

The largest drop occurred in Turn 2 and Turn 3. Turn 2 scores ranged from 55.48 for DeepSeekV4Pro to 66.60 for ClaudeOpus47, reflecting difficulties in correctly internalizing the scenario requirements. Turn 3 scores ranged from 66.94 to 76.24, reflecting the difficulty of converting prior reasoning into valid executable outputs. Overall protocol scores ranged from 69.34 for DeepSeekV4Pro to 77.38 for ClaudeOpus47.

Strict protocol pass rate was 0.0% for all LLMs. This does not mean every run failed completely; rather, it indicates that no model consistently satisfied all strict checks across all turns. The strict checks are demanding because they require simultaneous compliance with formatting, scenario interpretation, model/hyperparameter consistency, and output validity.

### 4.4.2 Common protocol violations

The most common protocol violations involved scenario semantics. Ground-truth usage was especially poorly understood: GPT55 had 0.00% ground-truth understanding, KimiK26 2.03%, Gemini31Pro 6.71%, DeepSeekV4Pro 11.67%, ClaudeOpus47 17.57%, and Grok43 25.17%. Retraining logic was also inconsistent, ranging from 24.83% for Gemini31Pro to 70.20% for Grok43. Block-wise reasoning was the weakest scenario-specific concept, with rates between 0.00% and 8.11% across LLMs.

These violations explain many downstream implementation errors. When an LLM misunderstood whether ground truth should be used, it often produced forecasts with the wrong recursive or rolling update logic. When it misunderstood block-wise retraining, it frequently emitted a forecast vector with the wrong length. When it misunderstood retraining frequency, it either refit too often, causing computational timeouts, or did not refit when required, causing logical mismatch with the scenario.

### 4.4.3 Recovery behavior and failure recovery

The protocol included multiple turns intended to let models revise and stabilize their reasoning before code generation. The evidence suggests partial recovery. Early turns often established the correct dataset and target, and later turns often matched the requested model and hyperparameters at the textual level. Code model match rates were 100% for all LLMs, and code hyperparameter match rates were also very high, ranging from 98.99% for KimiK26 to 100% for most models.

However, recovery was mostly lexical rather than operational. Models could repeat the correct model name and hyperparameters, but still failed to implement the scenario mechanics. This is why model/hyperparameter compliance appears high while execution success and scenario understanding remain much lower. In other words, the interactive protocol improved surface consistency but did not fully resolve deeper procedural reasoning errors.

## 4.5 Code generation analysis

The code generation stage was the main bottleneck in the experiment. Most final outputs were Python scripts, but successful execution required satisfying several constraints simultaneously: reading the correct input file, splitting chronologically, implementing the specified scenario, producing the correct forecast length, and printing a parseable flat list.

### 4.5.1 Code success rate

Overall, 517 out of 924 Part 2 rows executed or parsed successfully. Success rates differed strongly by dataset and LLM. AirPassengers and IceCreamHeater were highly successful at 87.50% and 83.33%, respectively. ILINet was moderate at 43.52%. Temperature and ETTh1 were low at 21.30% and 13.33%, respectively.

By LLM, ClaudeOpus47 had the highest success rate at 68.24%, followed by GPT55 at 66.22% and Gemini31Pro at 63.76%. Grok43 and KimiK26 were close to 50%, while DeepSeekV4Pro achieved 41.67%. These results indicate that stronger reasoning or model scale does not automatically guarantee executable forecasting code.

### 4.5.2 Runtime errors and logical errors

Runtime and infrastructure failures were frequent. Temperature had 125 timeouts, corresponding to 57.87% of all Temperature runs. ETTh1 had 17 timeouts out of 60 runs, or 28.33%. ILINet had 40 timeouts, or 18.52%. In contrast, AirPassengers and IceCreamHeater had no timeout failures, reflecting their smaller data size and simpler seasonal structure.

Logical errors also appeared in output-length validation. Temperature had 19 forecast-length mismatches, AirPassengers had 4, IceCreamHeater had 4, and ILINet had 3. Forecast-length errors are particularly important because the script may execute without crashing but still violate the forecasting protocol. They often occurred in multi-step or block-wise scenarios where the LLM confused horizon length, test length, and block size.

Code-level errors included syntax errors, value errors, name errors, type errors, and tensor shape errors. Temperature had the broadest range of code errors, including 16 value errors, 4 syntax errors, 2 tensor-shape errors, 1 name error, and 1 type error. ETTh1 had 6 syntax errors, reflecting the difficulty of handling multivariate data and long sequences.

### 4.5.3 Output validity and execution robustness

Output validity failures included model-returned errors, unstructured text outputs, and invalid forecast lists. KimiK26 had the highest number of list outputs, with 40 list rows, but also had 6 unstructured text rows and 13 forecast-length mismatches. Grok43 had 12 model-returned errors and 1 text output. These patterns show a tradeoff: direct list outputs avoid runtime execution, but they can still fail when the list length or structure is invalid.

The most robust code outputs came from ClaudeOpus47, GPT55, and Gemini31Pro. ClaudeOpus47 had no syntax-error category and only 4 forecast-length mismatches. GPT55 had no model-output issues but had 12 script syntax errors. Gemini31Pro had a relatively balanced profile, with 25 timeouts, 8 code-error runs, and 4 model-output issues.

## 4.6 Qualitative failure mode analysis

The quantitative failures can be grouped into four qualitative categories: data misunderstanding, model selection mistakes, hyperparameter inconsistencies, and hallucinated or unstable reasoning.

### 4.6.1 Data misunderstanding

Data misunderstanding occurred when the LLM incorrectly inferred the target variable, feature structure, time index, or required data split. Multivariate datasets were particularly vulnerable. ETTh1 required handling multiple input columns while forecasting the OT target. Some generated scripts either treated the data as univariate or failed to preserve the feature-target relationship. ILINet also suffered from confusion around weekly seasonality and forecast horizon.

Data misunderstanding was less common for AirPassengers because the dataset is short, univariate, and has a single obvious target. This explains why AirPassengers achieved the highest execution success rate. The IceCreamHeater dataset is multivariate, but the monthly structure and shorter length made it more manageable than ETTh1 or ILINet.

### 4.6.2 Model selection mistakes

At the recommendation level, model selection was often plausible but sometimes operationally risky. SARIMA was frequently selected for datasets with strong seasonality, which is statistically reasonable. However, the selected SARIMA configurations were not always computationally suitable for generated-code execution. Temperature is the clearest case: yearly seasonality with seasonal period 365 led to repeated timeouts. The result is a mismatch between statistical plausibility and execution feasibility.

Deep-learning challenger choices also caused practical failures. LSTM, GRU, and TCN are valid time-series models, but generated scripts often lacked stable training design, adequate scaling, and correct tensor-shape handling. Consequently, the ChalDL branch had the worst average performance despite being a reasonable model category in theory.

### 4.6.3 Hyperparameter inconsistencies

Hyperparameter instability was visible from the zero-shot recommendation stage. Model-family consensus was often higher than exact hyperparameter consensus. For example, Temperature S4 had 43.3% model consensus but only 16.7% exact hyperparameter consensus. ETTh1 S1 had 70.0% model consensus but only 13.3% exact hyperparameter consensus. This instability propagated into Part 2 because executable code depends heavily on exact values such as seasonal period, lag length, hidden size, sequence length, and forecast horizon.

Some hyperparameters were too heavy for the evaluation environment, such as SARIMA with seasonal period 365. Others were structurally inconsistent, especially in deep-learning scripts where `seq_len`, `pred_len`, and `input_size` had to align with the dataset and scenario. When this alignment failed, tensor-shape errors or forecast-length mismatches occurred.

### 4.6.4 Hallucinated assumptions and unstable reasoning

Several failures can be interpreted as hallucinated assumptions. LLMs sometimes assumed the presence of columns, preprocessing steps, or external libraries without verifying them from the provided context. Others inserted generic modeling routines that did not match the scenario. For example, a model might describe rolling retraining but implement a single static forecast, or describe block-wise updating but output a vector of length equal to one block rather than the full test set.

Unstable reasoning was also visible across repeated runs. The same LLM could produce a valid script in one run and an invalid script in another, even under the same dataset and scenario. This instability is central to the experimental conclusion: LLMs can generate plausible forecasting code, but their outputs require validation, execution checks, and metric-based filtering before being trusted.

## 4.7 Comparative discussion of model behavior

The results show that LLM behavior depends jointly on model family, dataset complexity, and forecast protocol. The best-performing LLM overall was not always the best model for every dataset-scenario pair, and higher reasoning ability did not always translate into higher execution robustness.

### 4.7.1 Standard models versus reasoning models

ClaudeOpus47 and GPT55 achieved the strongest overall execution robustness, with success rates of 68.24% and 66.22%. Gemini31Pro was close behind at 63.76%. DeepSeekV4Pro, Grok43, and KimiK26 showed more mixed behavior. DeepSeekV4Pro produced several best-case results but had a lower success rate of 41.67%. Grok43 showed relatively better scenario understanding in some fields, such as retraining and ground-truth usage, but had more model-returned errors and lower aggregate execution success.

This suggests that reasoning-oriented behavior is not sufficient by itself. A useful forecasting assistant must combine reasoning with disciplined code generation, output formatting, and computational awareness. ClaudeOpus47 appears to provide the best balance in this experiment: it was not always the best in every dataset-scenario pair, but it produced the most reliable aggregate performance.

### 4.7.2 Simple datasets versus complex datasets

The contrast between simple and complex datasets is clear. AirPassengers and IceCreamHeater had high execution success and interpretable results. They have shorter histories, lower dimensionality, and clear monthly seasonality. LLM-generated scripts could usually implement these settings correctly.

By contrast, ETTh1, ILINet, and Temperature exposed limitations. ETTh1 is multivariate and long, making generated scripts prone to timeouts and shape errors. ILINet has weekly structure and longer horizon logic, leading to timeouts and other failures. Temperature is univariate but daily, and the yearly seasonal period created severe computational problems. Thus, dataset simplicity is not only about dimensionality; frequency, length, seasonality, and update protocol all affect execution feasibility.

### 4.7.3 Short-horizon versus long-horizon forecasting

Short-horizon settings were generally easier than long-horizon or multi-step settings. S1 and S2 had the highest success rates, at 64.53% and 58.26%. S3 had the lowest success rate at 40.43%, indicating that multi-step rolling forecasts are difficult for LLMs to implement correctly. S4 recovered to 60.43%, but performance remained sensitive to block-size handling.

The forecasting results suggest that LLMs can be useful for short-horizon or structurally simple forecasting workflows, especially when the output is validated automatically. However, for long-horizon, block-wise, or continuously retrained settings, generated code is not reliable without strong guardrails. A practical LLM forecasting protocol should therefore include execution timeouts, output-length checks, forecast validity checks, and fallback baselines.

## Chapter summary

This chapter shows that LLMs can generate useful forecasting workflows, but their reliability is highly conditional. Traditional baselines remain difficult to beat, especially simple naive baselines in rolling-update settings. Zero-shot recommendations are often plausible at the model-family level, but exact hyperparameter recommendations are unstable and sometimes computationally unsafe. Interactive forecasting improves structure and allows code generation, but execution success varies sharply by dataset and scenario.

The main empirical conclusions are:

1. Traditional baselines are strong and heterogeneous; no single baseline dominates all scenarios.
2. LLM zero-shot recommendations favor statistically plausible models, especially SARIMA and Exponential Smoothing, but may ignore execution feasibility.
3. Interactive forecasting succeeds on simpler datasets but struggles on long, multivariate, or high-frequency seasonal datasets.
4. ChalML is the most executable challenger branch, but it does not outperform the Base branch on average.
5. ChalDL is the least reliable branch, with severe average performance degradation.
6. Protocol compliance is strong in early descriptive turns but weak in scenario-specific reasoning.
7. Code execution checks are essential because many scripts are plausible but invalid, too slow, or logically inconsistent.

These findings motivate the need for stricter constrained generation, automated validation, computationally aware hyperparameter selection, and fallback strategies when applying LLMs to time-series forecasting.
