# Corrected DistilBERT Routed Policy Ablation vs Gemini 3 Flash Baseline

## Main Ablation Table

| Model | Benchmark | N | Scored N | Official score mean | Judge mean | Mean Latency | Total Cost | Total Tokens | Failures/Unscored |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single Gemini 3 Flash baseline | overall | 1,371 | 1,371 | 0.7223 | 0.8600 | 9.53s | $6.3735 | 9,554,642 | 145/0 |
| google/gemini-3-flash-preview | mmlu_medical | 871 | 871 | 0.9242 | N/A | 1.27s | $0.0837 | 117,531 | 66/0 |
| google/gemini-3-flash-preview | healthbench | 400 | 400 | 0.3907 | N/A | 28.32s | $5.8501 | 8,379,424 | 8/0 |
| google/gemini-3-flash-preview | medagentbench | 100 | 100 | 0.2900 | 0.8600 | 6.29s | $0.4398 | 1,057,687 | 71/0 |
| Corrected DistilBERT Routed Policy | overall | 1,371 | 1,363 | 0.7001 | 0.7957 | 10.76s | $6.2448 | 10,932,155 | 171/8 |
| routed | mmlu_medical | 871 | 871 | 0.8840 | N/A | 0.50s | $0.0000 | 348,477 | 101/0 |
| routed | healthbench | 400 | 399 | 0.3789 | N/A | 34.60s | $6.1935 | 9,599,171 | 10/1 |
| routed | medagentbench | 100 | 93 | 0.3548 | 0.7957 | 4.76s | $0.0513 | 984,507 | 60/7 |

## Corrected Run Minus Baseline

| Benchmark | Official score delta | Judge mean delta | Mean latency delta | Cost delta | Token delta | Failure delta | Unscored delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | -0.0223 | -0.0643 | 1.23s | $-0.1288 | 1,377,513 | 26 | 8 |
| mmlu_medical | -0.0402 | N/A | -0.77s | $-0.0837 | 230,946 | 35 | 0 |
| healthbench | -0.0118 | N/A | 6.27s | $0.3435 | 1,219,747 | 2 | 1 |
| medagentbench | 0.0648 | -0.0643 | -1.53s | $-0.3885 | -73,180 | -11 | 7 |
