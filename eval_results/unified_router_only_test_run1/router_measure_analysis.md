# Router Measure Analysis: router_only

Source episodes: `eval_results/unified_router_only_test_run1/corrected_combined_episodes.jsonl`

## Headline Metrics

| Measure | Value |
| --- | --- |
| Episodes | 1371 |
| Mean reward | 0.7001 |
| Prompt-label accuracy | 95.7% |
| Tool-label accuracy | 92.6% |
| Model changed from default | 74.9% |
| Memory actually injected | 0.0% |

## Benchmark Breakdown

| Benchmark | n | Prompt accuracy | Tool accuracy | Actual FHIR prompt injection | Mean reward |
| --- | --- | --- | --- | --- | --- |
| healthbench | 400 | 90.5% | 99.8% | 0.0% | 0.3789 |
| medagentbench | 100 | 79.0% | 0.0% | 100.0% | 0.3548 |
| mmlu_medical | 871 | 100.0% | 100.0% | 0.0% | 0.8840 |

## Tool Routing

| Measure | Value |
| --- | --- |
| MedAgentBench selected `fhir` | 0.0% |
| Non-MedAgentBench selected `fhir` | 0.0% |
| MedAgentBench actual FHIR prompt injected | 100.0% |
| Non-MedAgentBench actual FHIR prompt injected | 0.0% |
| Tool confidence when correct | 0.9850 |
| Tool confidence when wrong | 0.7691 |

## Prompt Routing

| Measure | Value |
| --- | --- |
| Prompt confidence when correct | 0.8782 |
| Prompt confidence when wrong | 0.5549 |
| Mean reward when prompt label correct | 0.7177 |
| Mean reward when prompt label wrong | 0.2892 |