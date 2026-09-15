# Scenario compliance summary

This analysis evaluates whether an LLM can understand and execute the forecasting scenario required by the prompt. The unit of analysis is the run, and every metric is computed from explicit numerator/denominator definitions.

## 1. Metric definitions

For run $i$, let $R_i$ be the set of supported requirements for that scenario, as inferred from prompts/4scenarios.txt. For each requirement $r \in R_i$, define

- $S_{i,r} = 1$ if Turn 2 correctly matches the expected value, and $0$ otherwise.
- $I_{i,r} = 1$ if Turn 3 correctly matches the expected value, and $0$ otherwise.
- $SemanticCompliance_i = 100 	imes rac{\sum_{r \in R_i} S_{i,r}}{|R_i|}$.
- $ImplementationCompliance_i = 100 	imes rac{\sum_{r \in R_i} I_{i,r}}{|R_i|}$.
- $FullySemantic_i = 1[SemanticCompliance_i = 100]$.
- $FullyImplementation_i = 1[ImplementationCompliance_i = 100]$.
- $ExecutionSuccess_i = 1$ if the run is marked OK, and $0$ otherwise.

For a scenario $s$, let $n_s$ be the number of runs in that scenario. The scenario-level metrics are

- $AverageSemanticCompliance_s = rac{1}{n_s} \sum_i SemanticCompliance_i$.
- $AverageImplementationCompliance_s = rac{1}{n_s} \sum_i ImplementationCompliance_i$.
- $FullySemanticRuns_s = rac{1}{n_s} \sum_i FullySemantic_i$.
- $FullyImplementationRuns_s = rac{1}{n_s} \sum_i FullyImplementation_i$.
- $ImplementationGivenSemantic_s = rac{\sum_i 1[FullySemantic_i = 1 \land FullyImplementation_i = 1]}{\sum_i 1[FullySemantic_i = 1]}$.
- $ExecutionGivenImplementation_s = rac{\sum_i 1[FullyImplementation_i = 1 \land ExecutionSuccess_i = 1]}{\sum_i 1[FullyImplementation_i = 1]}$.
- Requirement-level Turn 2 correctness: $rac{\sum_i S_{i,r}}{n_s}$.
- Requirement-level Turn 3 correctness: $rac{\sum_i I_{i,r}}{n_s}$.
- Requirement-level scenario success: $rac{\sum_i 1[S_{i,r}=1 \land I_{i,r}=1]}{n_s}$.

## 2. Scenario-to-parser mapping

| Scenario | Requirement | Expected value | Parser field | Turn | Evaluation rule |
| --- | --- | --- | --- | --- | --- |
| S1 | Retraining | False | t2_understands_retraining / t3_retraining_implemented | Turn 2 / Turn 3 | semantic/implementation field == expected value |
| S1 | Ground truth | False | t2_understands_groundtruth / t3_uses_ground_truth | Turn 2 / Turn 3 | semantic/implementation field == expected value |
| S2 | Retraining | True | t2_understands_retraining / t3_retraining_implemented | Turn 2 / Turn 3 | semantic/implementation field == expected value |
| S2 | Ground truth | True | t2_understands_groundtruth / t3_uses_ground_truth | Turn 2 / Turn 3 | semantic/implementation field == expected value |
| S3 | Retraining | True | t2_understands_retraining / t3_retraining_implemented | Turn 2 / Turn 3 | semantic/implementation field == expected value |
| S3 | Ground truth | True | t2_understands_groundtruth / t3_uses_ground_truth | Turn 2 / Turn 3 | semantic/implementation field == expected value |
| S4 | Retraining | True | t2_understands_retraining / t3_retraining_implemented | Turn 2 / Turn 3 | semantic/implementation field == expected value |
| S4 | Ground truth | True | t2_understands_groundtruth / t3_uses_ground_truth | Turn 2 / Turn 3 | semantic/implementation field == expected value |
| S4 | Blockwise | True | t2_understands_blockwise / t3_correct_block_size | Turn 2 / Turn 3 | semantic/implementation field == expected value |

## 3. Scenario-level results

| Scenario | Runs | Avg semantic compliance | Avg implementation compliance | Fully semantic compliant | Fully implementation compliant | Execution success | Implementation given semantic | Execution given implementation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| S1 | 230 | 86.1% | 68.5% | 168 / 230 (73.0%) | 104 / 230 (45.2%) | 151 / 230 (65.7%) | 71 / 168 (42.3%) | 62 / 104 (59.6%) |
| S2 | 230 | 68.3% | 51.3% | 101 / 230 (43.9%) | 41 / 230 (17.8%) | 134 / 230 (58.3%) | 24 / 101 (23.8%) | 34 / 41 (82.9%) |
| S3 | 230 | 57.4% | 55.4% | 77 / 230 (33.5%) | 54 / 230 (23.5%) | 97 / 230 (42.2%) | 21 / 77 (27.3%) | 28 / 54 (51.9%) |
| S4 | 230 | 66.0% | 72.6% | 63 / 230 (27.4%) | 96 / 230 (41.7%) | 153 / 230 (66.5%) | 23 / 63 (36.5%) | 67 / 96 (69.8%) |

## 4. Requirement-level results

| Scenario | Requirement | Expected value | Supported | Turn 2 correctness | Turn 3 correctness | Scenario success |
| --- | --- | --- | --- | ---: | ---: | ---: |
| S1 | Retraining | False | Yes | 219 / 230 (95.2%) | 205 / 230 (89.1%) | 195 / 230 (84.8%) |
| S1 | Ground truth usage | False | Yes | 177 / 230 (77.0%) | 110 / 230 (47.8%) | 80 / 230 (34.8%) |
| S1 | Blockwise evaluation | — | No | n/a | n/a | n/a |
| S2 | Retraining | True | Yes | 176 / 230 (76.5%) | 74 / 230 (32.2%) | 60 / 230 (26.1%) |
| S2 | Ground truth usage | True | Yes | 138 / 230 (60.0%) | 162 / 230 (70.4%) | 99 / 230 (43.0%) |
| S2 | Blockwise evaluation | — | No | n/a | n/a | n/a |
| S3 | Retraining | True | Yes | 155 / 230 (67.4%) | 98 / 230 (42.6%) | 65 / 230 (28.3%) |
| S3 | Ground truth usage | True | Yes | 109 / 230 (47.4%) | 157 / 230 (68.3%) | 71 / 230 (30.9%) |
| S3 | Blockwise evaluation | — | No | n/a | n/a | n/a |
| S4 | Retraining | True | Yes | 167 / 230 (72.6%) | 126 / 230 (54.8%) | 96 / 230 (41.7%) |
| S4 | Ground truth usage | True | Yes | 114 / 230 (49.6%) | 194 / 230 (84.3%) | 97 / 230 (42.2%) |
| S4 | Blockwise evaluation | True | Yes | 174 / 230 (75.7%) | 181 / 230 (78.7%) | 135 / 230 (58.7%) |

## 5. Unsupported requirements

The current parser does not expose a dedicated field for multi-step horizon correctness. This requirement is therefore reported as unsupported rather than inferred from unrelated fields.

## 6. Verification samples

The script also writes a verification sample file at results/part2-interactive-llm-forecasting/analysis/scenario_compliance_verification_samples.txt. Each entry reports the scenario, expected requirements, Turn 2 values, Turn 3 values, semantic score, implementation score, execution status, and final classification.

