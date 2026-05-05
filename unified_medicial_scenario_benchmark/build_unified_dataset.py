"""Build and optionally upload bdanko/unified_medical_scenario_benchmark.

Sources:
- MMLU-Medical: cais/mmlu, medical subject configs and available splits.
- HealthBench: openai/healthbench full and hard JSONL files.
- MedAgentBench: bdanko/medagentbench splits created by build_medagentbench_hf.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, load_dataset

DEFAULT_SEED = 15179996
DEFAULT_REPO_ID = "bdanko/unified_medical_scenario_benchmark"
ROOT = Path(__file__).resolve().parents[1]
HEALTHBENCH_FULL = ROOT / "healthbench_cube" / "data" / "healthbench.jsonl"
HEALTHBENCH_HARD = ROOT / "healthbench_cube" / "data" / "hard_2025-05-08-21-00-10.jsonl"

MMLU_MEDICAL_SUBJECTS = [
    "anatomy",
    "clinical_knowledge",
    "college_biology",
    "college_medicine",
    "medical_genetics",
    "professional_medicine",
    "nutrition",
    "virology",
    "high_school_biology",
]


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    return hashlib.sha256(json_dumps(value).encode("utf-8")).hexdigest()


def empty_row() -> dict[str, Any]:
    return {
        "id": "",
        "benchmark": "",
        "source_dataset": "",
        "source_config": "",
        "source_split": "",
        "source_id": "",
        "source_index": -1,
        "scenario_type": "",
        "task_family": "",
        "question": "",
        "instruction": "",
        "context": "",
        "messages_json": "[]",
        "choices_json": "[]",
        "correct_option": "",
        "correct_answer": "",
        "acceptable_answers_json": "[]",
        "rubrics_json": "[]",
        "ideal_completions_json": "[]",
        "tools_json": "[]",
        "tool_names_json": "[]",
        "is_hard": False,
        "reward_spec_json": "{}",
        "eval_spec_json": "{}",
        "router_metadata_json": "{}",
        "original_record_sha256": "",
        "original_record_json": "{}",
    }


def normalize_mmlu_medical() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject in MMLU_MEDICAL_SUBJECTS:
        ds = load_dataset("cais/mmlu", subject)
        for split, split_ds in ds.items():
            for i, rec in enumerate(split_ds):
                row = empty_row()
                row.update(
                    {
                        "id": f"mmlu_medical::{subject}::{split}::{i}",
                        "benchmark": "mmlu_medical",
                        "source_dataset": "cais/mmlu",
                        "source_config": subject,
                        "source_split": split,
                        "source_id": f"{subject}:{split}:{i}",
                        "source_index": i,
                        "scenario_type": "multiple_choice_medical_knowledge",
                        "task_family": subject,
                        "question": rec["question"],
                        "choices_json": json_dumps(rec["choices"]),
                        "correct_option": "ABCD"[int(rec["answer"])],
                        "correct_answer": rec["choices"][int(rec["answer"])],
                        "reward_spec_json": json_dumps(
                            {
                                "type": "exact_match",
                                "reward": 1.0,
                                "penalty": 0.0,
                                "answer_action": "answer",
                                "answer_argument": "content",
                                "target": "correct_option",
                            }
                        ),
                        "eval_spec_json": json_dumps({"grader": "multiple_choice_exact_match"}),
                        "router_metadata_json": json_dumps({"expected_route_complexity": "low"}),
                        "original_record_sha256": stable_hash(rec),
                        "original_record_json": json_dumps(rec),
                    }
                )
                rows.append(row)
    return rows


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_healthbench_files() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    try:
        full = load_dataset(
            "json",
            data_files={"full": "hf://datasets/openai/healthbench/2025-05-07-06-14-12_oss_eval.jsonl"},
            split="full",
        )
        hard = load_dataset(
            "json",
            data_files={"hard": "hf://datasets/openai/healthbench/hard_2025-05-08-21-00-10.jsonl"},
            split="hard",
        )
        return list(full), list(hard)
    except Exception:
        return read_jsonl(HEALTHBENCH_FULL), read_jsonl(HEALTHBENCH_HARD)


def normalize_healthbench(seed: int) -> dict[str, list[dict[str, Any]]]:
    full, hard = load_healthbench_files()
    hard_ids = {rec["prompt_id"] for rec in hard}
    hard_by_id = {rec["prompt_id"]: rec for rec in hard}
    hard_rows = [hard_by_id[rec["prompt_id"]] for rec in full if rec["prompt_id"] in hard_by_id]
    if (len(full), len(hard_rows)) != (5000, 1000):
        raise ValueError(
            f"Unexpected HealthBench counts: full={len(full)} hard={len(hard_rows)}"
        )

    rng = random.Random(seed)
    rng.shuffle(hard_rows)
    split_plan = {
        "train": hard_rows[:400],
        "validation": hard_rows[400:600],
        "test": hard_rows[600:],
    }

    out: dict[str, list[dict[str, Any]]] = {}
    for split, records in split_plan.items():
        rows = []
        for i, rec in enumerate(records):
            is_hard = rec["prompt_id"] in hard_ids
            row = empty_row()
            row.update(
                {
                    "id": f"healthbench::{split}::{rec['prompt_id']}",
                    "benchmark": "healthbench",
                    "source_dataset": "openai/healthbench",
                    "source_split": "hard",
                    "source_id": str(rec["prompt_id"]),
                    "source_index": i,
                    "scenario_type": "rubric_graded_health_conversation",
                    "task_family": "hard" if is_hard else "regular",
                    "messages_json": json_dumps(rec["prompt"]),
                    "rubrics_json": json_dumps(rec["rubrics"]),
                    "ideal_completions_json": json_dumps(rec.get("ideal_completions_data", [])),
                    "is_hard": is_hard,
                    "reward_spec_json": json_dumps(
                        {
                            "type": "weighted_rubric",
                            "range": [0.0, 1.0],
                            "penalty": 0.0,
                            "answer_action": "answer",
                            "answer_argument": "content",
                            "score": "sum(points for met positive criteria) / sum(points for positive criteria)",
                        }
                    ),
                    "eval_spec_json": json_dumps(
                        {
                            "grader": "rubric_llm_grader",
                            "rubric_field": "rubrics_json",
                            "grader_template": "healthbench_cube.GRADER_TEMPLATE compatible",
                        }
                    ),
                    "router_metadata_json": json_dumps(
                        {
                            "expected_route_complexity": "high",
                            "guardrail_emphasis": True,
                            "example_tags": rec.get("example_tags", []),
                            "canary": rec.get("canary", ""),
                        }
                    ),
                    "original_record_sha256": stable_hash(rec),
                    "original_record_json": json_dumps(rec),
                }
            )
            rows.append(row)
        out[split] = rows
    return out


def normalize_medagentbench(repo_id: str) -> dict[str, list[dict[str, Any]]]:
    ds = load_dataset(repo_id)
    out: dict[str, list[dict[str, Any]]] = {}
    for split, split_ds in ds.items():
        rows = []
        for i, rec in enumerate(split_ds):
            row = empty_row()
            row.update(
                {
                    "id": f"medagentbench::{split}::{rec['source_id']}",
                    "benchmark": "medagentbench",
                    "source_dataset": repo_id,
                    "source_split": split,
                    "source_id": rec["source_id"],
                    "source_index": i,
                    "scenario_type": "tool_using_ehr_agent_task",
                    "task_family": rec["task_family"],
                    "instruction": rec["instruction"],
                    "context": rec.get("context", ""),
                    "acceptable_answers_json": rec["acceptable_answers_json"],
                    "tools_json": rec["tools_json"],
                    "tool_names_json": rec["tool_names_json"],
                    "reward_spec_json": rec["reward_spec_json"],
                    "eval_spec_json": rec["eval_spec_json"],
                    "router_metadata_json": json_dumps(
                        {"expected_route_complexity": "agentic", "requires_tool_selection": True}
                    ),
                    "original_record_sha256": rec["original_record_sha256"],
                    "original_record_json": rec["original_record_json"],
                }
            )
            rows.append(row)
        out[split] = rows
    return out


def split_mmlu_medical(rows: list[dict[str, Any]], seed: int) -> dict[str, list[dict[str, Any]]]:
    test_rows = [row for row in rows if row["source_split"] == "test"]
    if len(test_rows) != 1871:
        raise ValueError(f"Expected 1871 MMLU-Medical test rows, found {len(test_rows)}")
    shuffled = list(test_rows)
    random.Random(seed).shuffle(shuffled)
    return {
        "train": shuffled[:800],
        "validation": shuffled[800:1000],
        "test": shuffled[1000:],
    }


def ds_from_rows(rows: Iterable[dict[str, Any]]) -> Dataset:
    return Dataset.from_list(list(rows))


def target_split(source_split: str) -> str:
    if source_split == "dev":
        return "validation"
    if source_split in {"train", "validation", "test"}:
        return source_split
    raise ValueError(f"Unsupported source split: {source_split}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--medagentbench-repo-id", default="bdanko/medagentbench")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default="hf_out/unified_medical_scenario_benchmark")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    mmlu_rows = normalize_mmlu_medical()
    health_rows = normalize_healthbench(args.seed)
    medagent_rows = normalize_medagentbench(args.medagentbench_repo_id)
    mmlu_split_rows = split_mmlu_medical(mmlu_rows, args.seed)

    split_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    for split, rows in mmlu_split_rows.items():
        split_rows[split].extend(rows)
    for split, rows in health_rows.items():
        split_rows[split].extend(rows)
    for split, rows in medagent_rows.items():
        split_rows[split].extend(rows)

    dataset = DatasetDict()
    for split, rows in split_rows.items():
        dataset[split] = Dataset.from_list(rows)

    dataset.save_to_disk(args.output_dir)
    print(dataset)
    print({split: dataset[split].num_rows for split in dataset})
    counts: dict[str, dict[str, int]] = {}
    for split, rows in split_rows.items():
        counts[split] = {}
        for row in rows:
            counts[split][row["benchmark"]] = counts[split].get(row["benchmark"], 0) + 1
    print(counts)
    if args.push:
        dataset.push_to_hub(args.repo_id, private=False)
        print(f"Pushed {args.repo_id}")


if __name__ == "__main__":
    main()
