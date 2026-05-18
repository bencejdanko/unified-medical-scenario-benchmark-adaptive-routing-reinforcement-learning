# Router Measure Analysis: router_memory

Source episodes: `eval_results/unified_router_memory_test_run1/corrected_combined_episodes.jsonl`

## Headline Metrics

| Measure | Value |
| --- | --- |
| Episodes | 1371 |
| Mean reward | 0.6845 |
| Prompt-label accuracy | 95.7% |
| Tool-label accuracy | 92.7% |
| Model changed from default | 75.0% |
| Memory actually injected | 100.0% |

## Benchmark Breakdown

| Benchmark | n | Prompt accuracy | Tool accuracy | Actual FHIR prompt injection | Mean reward |
| --- | --- | --- | --- | --- | --- |
| healthbench | 400 | 90.5% | 100.0% | 0.0% | 0.3615 |
| medagentbench | 100 | 79.0% | 0.0% | 100.0% | 0.2000 |
| mmlu_medical | 871 | 100.0% | 100.0% | 0.0% | 0.8829 |

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
| Prompt confidence when wrong | 0.5530 |
| Mean reward when prompt label correct | 0.7049 |
| Mean reward when prompt label wrong | 0.2082 |