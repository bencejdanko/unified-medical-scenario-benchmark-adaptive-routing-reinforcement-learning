# Latency Component 

## Component Means

| Run | Benchmark | N | Total Mean | Response Mean | Agent Mean | Judge Mean | Official Scorer Mean | Reset Mean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single Gemini 3 Flash baseline | mmlu_medical | 871 | 1.27s | N/A | N/A | N/A | N/A | N/A |
| Single Gemini 3 Flash baseline | healthbench | 400 | 28.32s | 10.23s | N/A | 17.93s | N/A | N/A |
| Single Gemini 3 Flash baseline | medagentbench | 100 | 6.29s | N/A | 4.08s | 1.95s | 0.27s | 0.00s |
| Corrected DistilBERT Routed Policy | mmlu_medical | 871 | 0.50s | N/A | N/A | N/A | N/A | N/A |
| Corrected DistilBERT Routed Policy | healthbench | 400 | 34.60s | 11.37s | N/A | 23.08s | N/A | N/A |
| Corrected DistilBERT Routed Policy | medagentbench | 100 | 4.76s | N/A | 2.90s | 1.63s | 0.23s | 0.00s |
| Corrected DistilBERT Routed Policy + Episodic Memory | mmlu_medical | 871 | 0.60s | N/A | N/A | N/A | N/A | N/A |
| Corrected DistilBERT Routed Policy + Episodic Memory | healthbench | 400 | 33.50s | 10.73s | N/A | 22.55s | N/A | N/A |
| Corrected DistilBERT Routed Policy + Episodic Memory | medagentbench | 100 | 4.97s | N/A | 3.05s | 1.61s | 0.31s | 0.00s |


### DistilBERT Routed Policy

| Benchmark | Total Delta | Response/Agent Delta | Judge Delta | Official Scorer Delta |
| --- | ---: | ---: | ---: | ---: |
| mmlu_medical | -0.77s | N/A | N/A | N/A |
| healthbench | +6.27s | +1.14s | +5.15s | N/A |
| medagentbench | -1.53s | -1.17s | -0.32s | -0.04s |

### DistilBERT Routed Policy + Episodic Memory

| Benchmark | Total Delta | Response/Agent Delta | Judge Delta | Official Scorer Delta |
| --- | ---: | ---: | ---: | ---: |
| mmlu_medical | -0.67s | N/A | N/A | N/A |
| healthbench | +5.17s | +0.50s | +4.61s | N/A |
| medagentbench | -1.32s | -1.02s | -0.34s | +0.04s |

## Local No-OpenRouter Routing Overhead

| Mode | Mean | Median | P95 | P99 | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Router only | 99 ms | 85 ms | 196 ms | 273 ms | 530 ms |
| Router + memory, router component | 162 ms | 158 ms | 257 ms | 331 ms | 392 ms |
| Router + memory, memory retrieval | 208 ms | 180 ms | 390 ms | 609 ms | 844 ms |
| Router + memory, combined | 370 ms | 332 ms | 635 ms | 959 ms | 1174 ms |

By benchmark, mean local overhead was:

| Benchmark | Router Only Mean | Router + Memory Combined Mean |
| --- | ---: | ---: |
| mmlu_medical | 97 ms | 381 ms |
| healthbench | 97 ms | 341 ms |
| medagentbench | 129 ms | 398 ms |
