# Content-Only DistilBERT Router Training Report

## Training Run

- Modal run: `ap-RlVKdwTbotKFpYr0zvmzCw`
- Dataset: `bdanko/umsb-routing-classification-content-only`
- Uploaded model: `bdanko/umsb-distilbert-router-content-only`
- Base model: `distilbert/distilbert-base-uncased`
- Architecture: DistilBERT encoder with three classification heads
  - model route head
  - tool route head
  - prompt route head
- Epochs: 4
- Batch size: 32
- Learning rate: `2e-5`
- Max sequence length: 384 tokens

## Input and Output

Input to the model is one compact text string per task. The content-only dataset removes explicit benchmark metadata such as `benchmark`, `scenario_type`, `task_family`, and `is_hard`.

Example input shape:

```text
Task:
What’s the last HbA1C value in the chart for patient S2154941 and when was it recorded?

Context:
It's 2023-11-13T10:15:00+00:00 now...
```

Outputs are three classification labels:

| Head | Output Type | Labels |
| --- | --- | --- |
| Model | multi-class | `google/gemini-3-flash-preview`, `openai/gpt-oss-120b:nitro`, `google/gemma-4-31b-it` |
| Tool | binary | `none`, `fhir` |
| Prompt | multi-class | `mmlu_medical_json`, `healthbench_default`, `medagentbench_fhir` |

## Split Metrics

| Split | n | Model Accuracy | Tool Accuracy | Prompt Accuracy | Joint Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 1350 | 99.93% | 100.00% | 99.85% | 99.85% |
| Validation | 450 | 100.00% | 100.00% | 100.00% | 100.00% |
| Test | 1371 | 99.93% | 100.00% | 99.93% | 99.93% |

Test macro-F1:

| Head | Macro-F1 |
| --- | ---: |
| Model | 99.79% |
| Tool | 100.00% |
| Prompt | 99.79% |

## Test Confusion Matrices

Model labels:

```text
0 = google/gemini-3-flash-preview
1 = openai/gpt-oss-120b:nitro
2 = google/gemma-4-31b-it
```

Model confusion matrix:

```text
true\pred    0    1    2
0          399    0    1
1            0  871    0
2            0    0  100
```

Tool labels:

```text
0 = none
1 = fhir
```

Tool confusion matrix:

```text
true\pred     0    1
0          1271    0
1             0  100
```

Prompt labels:

```text
0 = mmlu_medical_json
1 = healthbench_default
2 = medagentbench_fhir
```

Prompt confusion matrix:

```text
true\pred    0    1    2
0          871    0    0
1            0  399    1
2            0    0  100
```