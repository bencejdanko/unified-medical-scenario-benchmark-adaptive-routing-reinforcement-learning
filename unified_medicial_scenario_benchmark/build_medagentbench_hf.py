"""Build and optionally upload a tabular MedAgentBench dataset to Hugging Face.

The local MedAgentBench implementation stores task cases as JSON plus a shared
FHIR tool schema. This script converts that source into efficient tabular
DatasetDict splits suitable for `datasets.load_dataset("bdanko/medagentbench")`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict

DEFAULT_SEED = 15179996
DEFAULT_REPO_ID = "bdanko/medagentbench"
ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "medagentbench_cube" / "data" / "medagentbench" / "test_data_v2.json"
TOOLS_PATH = ROOT / "medagentbench_cube" / "data" / "medagentbench" / "funcs_v1.json"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def derive_tool_name(tool_def: dict[str, Any]) -> str:
    path = (
        tool_def["name"]
        .replace("GET ", "get_")
        .replace("POST ", "post_")
        .replace("{api_base}/", "")
        .replace("/", "_")
    )
    desc = tool_def["description"].lower()
    if "labs" in desc:
        suffix = "_labs"
    elif "vitals" in desc:
        suffix = "_vitals"
    elif "problems" in desc:
        suffix = "_problems"
    elif "orders" in desc:
        suffix = "_orders"
    else:
        suffix = ""
    return f"{path}{suffix}" if suffix and not path.endswith(suffix) else path


def task_family(task_id: str) -> str:
    return task_id.split("_", 1)[0]


def load_rows(data_path: Path = DATA_PATH, tools_path: Path = TOOLS_PATH) -> list[dict[str, Any]]:
    cases = json.loads(data_path.read_text())
    tools = json.loads(tools_path.read_text())
    tool_names = [derive_tool_name(t) for t in tools]
    tools_json = json_dumps(tools)
    rows: list[dict[str, Any]] = []
    for i, case in enumerate(cases):
        source_id = str(case["id"])
        acceptable_answers = case.get("sol") or []
        eval_fields = {k: v for k, v in case.items() if k.startswith("eval_")}
        rows.append(
            {
                "id": f"medagentbench::{source_id}",
                "source_dataset": "bdanko/medagentbench",
                "source_local_path": str(data_path.relative_to(ROOT)),
                "source_id": source_id,
                "source_index": i,
                "task_family": task_family(source_id),
                "instruction": case["instruction"],
                "context": case.get("context", ""),
                "acceptable_answers_json": json_dumps(acceptable_answers),
                "eval_fields_json": json_dumps(eval_fields),
                "tools_json": tools_json,
                "tool_names_json": json_dumps(tool_names),
                "tool_count": len(tools),
                "reward_spec_json": json_dumps(
                    {
                        "type": "official_medagentbench_task_success",
                        "reward": 1.0,
                        "penalty": 0.0,
                        "agent_finish_format": "FINISH([answer1, answer2, ...])",
                        "official_metric": "pass@1 task success rate",
                    }
                ),
                "eval_spec_json": json_dumps(
                    {
                        "grader": "official_medagentbench_refsol",
                        "official_eval_entrypoint": "src.server.tasks.medagentbench.eval.eval",
                        "official_refsol_module": "src.server.tasks.medagentbench.refsol",
                        "refsol_distribution": "Download separately from the Stanford Box link in the official MedAgentBench README unless redistribution rights are confirmed.",
                        "requires_fhir_server": True,
                        "fhir_api_base_default": "http://localhost:8080/fhir/",
                        "tool_schema": "FHIR-like HTTP tool definitions from funcs_v1.json",
                        "notes": "This dataset includes task data and tool schemas only. Official scoring requires the external refsol.py plus a compatible FHIR server; do not use acceptable_answers_json as the benchmark grader.",
                    }
                ),
                "original_record_sha256": hashlib.sha256(json_dumps(case).encode("utf-8")).hexdigest(),
                "original_record_json": json_dumps(case),
            }
        )
    return rows


def split_rows(rows: list[dict[str, Any]], seed: int) -> DatasetDict:
    shuffled = list(rows)
    random.Random(seed).shuffle(shuffled)
    split_sizes = {"train": 150, "validation": 50, "test": 100}
    if len(shuffled) != sum(split_sizes.values()):
        raise ValueError(f"Expected 300 MedAgentBench rows, found {len(shuffled)}")
    out: dict[str, Dataset] = {}
    start = 0
    for split, size in split_sizes.items():
        chunk = []
        for row in shuffled[start : start + size]:
            row = dict(row)
            row["split"] = split
            row["split_seed"] = seed
            chunk.append(row)
        out[split] = Dataset.from_list(chunk)
        start += size
    return DatasetDict(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output-dir", default="hf_out/medagentbench")
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    dataset = split_rows(load_rows(), args.seed)
    dataset.save_to_disk(args.output_dir)
    print(dataset)
    print({split: dataset[split].num_rows for split in dataset})
    if args.push:
        dataset.push_to_hub(args.repo_id, private=False)
        print(f"Pushed {args.repo_id}")


if __name__ == "__main__":
    main()
