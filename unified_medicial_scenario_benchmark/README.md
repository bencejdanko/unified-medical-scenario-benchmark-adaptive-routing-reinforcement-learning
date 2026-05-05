# Unified Medical Scenario Benchmark Builders

This folder builds two Hugging Face datasets:

- `bdanko/medagentbench`: a tabular, split version of the local 300-case MedAgentBench implementation.
- `bdanko/unified_medical_scenario_benchmark`: one CUBE-oriented table spanning MMLU-Medical, HealthBench, and MedAgentBench.

The fixed split seed is `15179996`.

## Sources

- MMLU-Medical: `cais/mmlu`, configs `anatomy`, `clinical_knowledge`, `college_biology`, `college_medicine`, `medical_genetics`, `professional_medicine`, `nutrition`, `virology`, and `high_school_biology`.
- HealthBench: `openai/healthbench`, full file `2025-05-07-06-14-12_oss_eval.jsonl` and hard file `hard_2025-05-08-21-00-10.jsonl`.
- MedAgentBench: local `medagentbench_cube/data/medagentbench/test_data_v2.json` and `funcs_v1.json`, uploaded first as `bdanko/medagentbench`.

## Build

```bash
python unified_medicial_scenario_benchmark/build_medagentbench_hf.py
python unified_medicial_scenario_benchmark/build_unified_dataset.py
```

Add `--push` to upload when authenticated with Hugging Face:

```bash
python unified_medicial_scenario_benchmark/build_medagentbench_hf.py --push
python unified_medicial_scenario_benchmark/build_unified_dataset.py --push
```

## Splits

HealthBench uses the requested split policy:

- Train: 400 hard.
- Validation: 200 hard.
- Test: 400 hard.

MMLU-Medical uses 800 train, 200 validation, and 871 test rows sampled from the 1,871-question medical test pool.

MedAgentBench uses 150 train, 50 validation, and 100 test rows.

All rows expose shared CUBE-facing columns including `benchmark`, `scenario_type`, `question`, `instruction`, `messages_json`, `choices_json`, `rubrics_json`, `tools_json`, `reward_spec_json`, `eval_spec_json`, `router_metadata_json`, and `original_record_json`.

For MedAgentBench, `reward_spec_json` and `eval_spec_json` point to the official MedAgentBench reference-solution evaluator. The external `refsol.py` file is not vendored here; obtain it through the official MedAgentBench distribution unless redistribution rights are confirmed.
