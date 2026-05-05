---
license: mit
task_categories:
- question-answering
- text-generation
language:
- en
pretty_name: Unified Medical Scenario Benchmark
size_categories:
- 1K<n<10K
tags:
- medical
- benchmark
- cube
- contextual-bandit
- tool-use
- healthbench
- mmlu
- medagentbench
---

# Unified Medical Scenario Benchmark

This dataset unifies three medical AI evaluation sources into a single CUBE-oriented tabular schema for future prompt/model/tool routing experiments:

- **MMLU-Medical** from `cais/mmlu`, using medical subject configs only.
- **HealthBench** from `openai/healthbench`, using the full 5,000-example set plus the 1,000-example hard subset marker.
- **MedAgentBench** from `bdanko/medagentbench`, converted from the local `medagentbench_cube` implementation.

MedQA is intentionally excluded from this unified dataset. The intended simple knowledge-routing source is MMLU-Medical because its questions are more diverse in structure.

## Splits

Random seed: `15179996`.

| Split | Total | MMLU-Medical | HealthBench | MedAgentBench |
|---|---:|---:|---:|---:|
| train | 1,350 | 800 | 400 | 150 |
| validation | 450 | 200 | 200 | 50 |
| test | 1,371 | 871 | 400 | 100 |

HealthBench split policy:

- Train: 400 hard.
- Validation: 200 hard.
- Test: 400 hard.

MedAgentBench split policy:

- Train: 150.
- Validation: 50.
- Test: 100.

MMLU-Medical split policy:

- Train: 800.
- Validation: 200.
- Test: 871.

MMLU-Medical rows are sampled from the 1,871-question medical test pool using the fixed seed above.

## Schema

The dataset uses one shared schema across all benchmark families. JSON-bearing fields are encoded as strings for stable Parquet compatibility.

Important columns:

- `benchmark`: `mmlu_medical`, `healthbench`, or `medagentbench`.
- `source_dataset`, `source_config`, `source_split`, `source_id`: provenance.
- `scenario_type`: routing-relevant task category.
- `question`, `choices_json`, `correct_option`, `correct_answer`: multiple-choice evaluation fields.
- `messages_json`, `rubrics_json`, `ideal_completions_json`: HealthBench conversation and rubric fields.
- `instruction`, `context`, `acceptable_answers_json`, `tools_json`, `tool_names_json`: MedAgentBench agent task fields.
- `is_hard`: HealthBench hard subset marker.
- `reward_spec_json`: machine-readable reward contract.
- `eval_spec_json`: machine-readable evaluation/grader contract.
- `router_metadata_json`: routing metadata such as complexity and guardrail/tool-use hints.
- `original_record_sha256`, `original_record_json`: source-record audit trail.

## Reward Semantics

MMLU-Medical uses exact-match multiple-choice reward:

- Reward `1.0` when the answer action's `content` matches `correct_option`.
- Reward `0.0` otherwise.

HealthBench uses weighted rubric reward:

- Score range `[0.0, 1.0]`.
- Score is based on physician-authored rubric criteria.
- Evaluation requires a rubric grader compatible with the HealthBench grader template.

MedAgentBench uses the official MedAgentBench task success evaluator:

- Reward `1.0` when the official evaluator returns task success.
- Reward `0.0` otherwise.
- Evaluation requires a compatible FHIR server and the official `src/server/tasks/medagentbench/refsol.py` module distributed separately by the MedAgentBench authors.
- The tabular dataset includes task data and tool schemas. It does not use `acceptable_answers_json` as the benchmark grader.

## Load Example

```python
from datasets import load_dataset

ds = load_dataset("bdanko/unified_medical_scenario_benchmark")
print(ds)
```
