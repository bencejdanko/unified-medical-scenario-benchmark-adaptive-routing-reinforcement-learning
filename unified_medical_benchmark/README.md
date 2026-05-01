# Unified Medical Scenario Benchmark

This package composes the existing CUBE-wrapped medical benchmarks into one benchmark substrate for adaptive medical routing research.

The goal is not to prescribe a router or evaluation policy. The goal is to expose a consistent benchmark surface where each task carries enough metadata and trace information for future routers to decide whether a request is simple QA, evidence-grounded QA, rubric-scored clinical conversation, or EHR tool-use work.

## Architecture

- **Native CUBE wrappers remain authoritative**
  - MedQA, MedMCQA, PubMedQA, MMLU-Medical, HealthBench, and MedAgentBench keep their existing task reset/step/reward behavior.
  - The unified benchmark delegates to those tasks rather than duplicating scoring logic.

- **`BenchmarkAdapter` maps each source benchmark into a shared scenario taxonomy**
  - `scenario_type`: e.g. `exam_qa`, `literature_qa`, `rubric_conversation`, `ehr_tool_use`
  - `clinical_domain`: broad medical domain for routing-aware analysis
  - `complexity`: coarse complexity label intended for curriculum and routing experiments
  - `answer_format`: expected output shape
  - `requires_tools` and `requires_rubric_grading`: operational requirements that matter for real-world routing

- **`UnifiedMedicalBenchmark` is the combined CUBE benchmark**
  - It builds selected source benchmarks from the registry.
  - It creates stable task IDs in the form `<benchmark>::<source_task_id>`.
  - It returns CUBE-compatible `UnifiedMedicalTask` instances.

- **`UnifiedMedicalTask` adds traceability**
  - `reset()` appends `scenario_metadata` to the observation.
  - `step()` delegates to the native task, preserves its reward, and adds `step_record` plus cumulative `step_records` into `EnvironmentOutput.info`.

## Usage

```python
from cube.core import Action
from unified_medical_benchmark import build_benchmark

bench = build_benchmark(
    benchmark_names=["medqa", "medmcqa", "pubmedqa", "mmlu_medical"],
    num_examples_per_benchmark=10,
)

task = bench.get_task(0)
obs, info = task.reset()
print(info["scenario_metadata"])

result = task.step(Action(name="answer", arguments={"content": "A"}))
print(result.reward)
print(result.info["step_records"])
```

## Including HealthBench and MedAgentBench

HealthBench and MedAgentBench are part of the default registry, but they have additional runtime needs.

- **HealthBench**
  - Loads local JSONL data from `healthbench_cube/data/healthbench.jsonl`.
  - Scoring free-text responses requires `OPENAI_API_KEY` because the current HealthBench CUBE wrapper uses an LLM grader.

- **MedAgentBench**
  - Defaults to local `test_data_v1.json` and `funcs_v1.json` under `medagentbench_cube/data/medagentbench`.
  - Tool calls require the MedAgentBench FHIR server to be running on `localhost:8080`.

## Why this is useful for routing

The unified benchmark makes routing-relevant differences explicit without implementing a router:

- **Complexity-aware selection** can use `complexity`, `scenario_type`, and `requires_tools`.
- **Category-aware evaluation** can stratify performance by `benchmark`, `clinical_domain`, and source-specific `category`.
- **Step-aware analysis** can inspect `step_records` to measure tool-use length, latency, reward trajectory, and failure modes.
- **CUBE compatibility** means existing CUBE runners can still call `reset()` and `step()` normally.

## Current scenario taxonomy

| Benchmark | Scenario type | Complexity | Operational requirement |
| --- | --- | --- | --- |
| MedQA | `exam_qa` | `moderate` | single-choice answer |
| MedMCQA | `exam_qa` | `moderate` | single-choice answer |
| PubMedQA | `literature_qa` | `moderate` | context-grounded yes/no/maybe answer |
| MMLU-Medical | `exam_qa` | `low_to_moderate` | single-choice answer |
| HealthBench | `rubric_conversation` | `high` | rubric-based LLM grading |
| MedAgentBench | `ehr_tool_use` | `high` | FHIR tool calls |
