# DistilBERT Routed Policy + Episodic Memory Ablation vs Gemini 3 Flash Baseline

## Main Ablation Table

| Model | Benchmark | N | Scored N | Official score mean | Judge mean | Mean Latency | Total Cost | Total Tokens | Failures/Unscored |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single Gemini 3 Flash baseline | overall | 1,371 | 1,371 | 0.7223 | 0.8600 | 9.53s | $6.3735 | 9,554,642 | 145/0 |
| google/gemini-3-flash-preview | mmlu_medical | 871 | 871 | 0.9242 | N/A | 1.27s | $0.0837 | 117,531 | 66/0 |
| google/gemini-3-flash-preview | healthbench | 400 | 400 | 0.3907 | N/A | 28.32s | $5.8501 | 8,379,424 | 8/0 |
| google/gemini-3-flash-preview | medagentbench | 100 | 100 | 0.2900 | 0.8600 | 6.29s | $0.4398 | 1,057,687 | 71/0 |
| Corrected DistilBERT Routed Policy + Episodic Memory | overall | 1,371 | 1,361 | 0.6845 | 0.8111 | 10.52s | $7.0274 | 13,002,474 | 183/10 |
| routed | mmlu_medical | 871 | 871 | 0.8829 | N/A | 0.60s | $0.0003 | 750,067 | 102/0 |
| routed | healthbench | 400 | 400 | 0.3615 | N/A | 33.50s | $6.9706 | 11,181,116 | 9/0 |
| routed | medagentbench | 100 | 90 | 0.2000 | 0.8111 | 4.97s | $0.0566 | 1,071,291 | 72/10 |

## Corrected Run Minus Baseline

| Benchmark | Official score delta | Judge mean delta | Mean latency delta | Cost delta | Token delta | Failure delta | Unscored delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | -0.0378 | -0.0489 | 0.99s | $0.6539 | 3,447,832 | 38 | 10 |
| mmlu_medical | -0.0413 | N/A | -0.67s | $-0.0835 | 632,536 | 36 | 0 |
| healthbench | -0.0293 | N/A | 5.17s | $1.1205 | 2,801,692 | 1 | 0 |
| medagentbench | -0.0900 | -0.0489 | -1.32s | $-0.3832 | 13,604 | 1 | 10 |

## Corrected Router + Memory Minus Corrected Router-Only

| Benchmark | Official score delta | Judge mean delta | Mean latency delta | Cost delta | Token delta | Failure delta | Unscored delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| overall | -0.0156 | 0.0154 | -0.24s | $0.7827 | 2,070,319 | 12 | 2 |
| mmlu_medical | -0.0011 | N/A | 0.10s | $0.0003 | 401,590 | 1 | 0 |
| healthbench | -0.0174 | N/A | -1.10s | $0.7771 | 1,581,945 | -1 | -1 |
| medagentbench | -0.1548 | 0.0154 | 0.21s | $0.0053 | 86,784 | 12 | 3 |