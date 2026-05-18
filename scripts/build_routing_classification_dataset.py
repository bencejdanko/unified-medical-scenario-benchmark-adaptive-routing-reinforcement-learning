"""Build and optionally upload the UMSB routing classification dataset.

The labels are policy-supervised from the intended first router:
  - mmlu_medical -> openai/gpt-oss-120b:nitro
  - medagentbench -> google/gemma-4-31b-it
  - healthbench -> google/gemini-3-flash-preview

The dataset is multi-head: model, tool, and prompt labels are stored as both
strings and integer ids so a DistilBERT classifier can train directly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset, load_from_disk


DEFAULT_SOURCE = "bdanko/unified_medical_scenario_benchmark"
DEFAULT_REPO = "bdanko/umsb-routing-classification"

MODEL_LABELS = [
    "google/gemini-3-flash-preview",
    "openai/gpt-oss-120b:nitro",
    "google/gemma-4-31b-it",
]
TOOL_LABELS = ["none", "fhir"]
PROMPT_LABELS = ["mmlu_medical_json", "healthbench_default", "medagentbench_fhir"]

MODEL_BY_BENCHMARK = {
    "mmlu_medical": "openai/gpt-oss-120b:nitro",
    "healthbench": "google/gemini-3-flash-preview",
    "medagentbench": "google/gemma-4-31b-it",
}
TOOL_BY_BENCHMARK = {
    "mmlu_medical": "none",
    "healthbench": "none",
    "medagentbench": "fhir",
}
PROMPT_BY_BENCHMARK = {
    "mmlu_medical": "mmlu_medical_json",
    "healthbench": "healthbench_default",
    "medagentbench": "medagentbench_fhir",
}


def parse_json_field(row: dict[str, Any], key: str, default: Any) -> Any:
    value = row.get(key)
    if value in (None, ""):
        return default
    try:
        return json.loads(value)
    except Exception:
        return default


def metadata_rich_router_text(row: dict[str, Any]) -> str:
    benchmark = row["benchmark"]
    parts = [
        f"benchmark: {benchmark}",
        f"scenario_type: {row.get('scenario_type', '')}",
        f"task_family: {row.get('task_family', '')}",
        f"is_hard: {row.get('is_hard', '')}",
    ]
    if benchmark == "mmlu_medical":
        choices = parse_json_field(row, "choices_json", [])
        choice_text = " ".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
        parts.append(f"question: {row.get('question', '')}")
        parts.append(f"choices: {choice_text}")
    elif benchmark == "healthbench":
        messages = parse_json_field(row, "messages_json", [])
        user_text = " ".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        metadata = parse_json_field(row, "router_metadata_json", {})
        parts.append(f"patient_or_user_request: {user_text}")
        parts.append(f"tags: {' '.join(metadata.get('example_tags', []))}")
    elif benchmark == "medagentbench":
        tools = parse_json_field(row, "tools_json", [])
        tool_names = [str(t.get("name") or t.get("function", {}).get("name") or "") for t in tools if isinstance(t, dict)]
        parts.append(f"context: {row.get('context', '')}")
        parts.append(f"instruction: {row.get('instruction', '')}")
        parts.append(f"available_tools: {' '.join(name for name in tool_names if name)}")
    return "\n".join(part for part in parts if part.strip())


def content_only_router_text(row: dict[str, Any]) -> str:
    """Build a realistic router input without explicit benchmark metadata.

    The goal is to resemble what a production router could see before choosing
    an execution path: the user's task content, optional clinical context, and
    answer options when the user supplied them. It intentionally omits
    benchmark, scenario_type, task_family, is_hard, labels, rubrics, and tool
    names because those fields leak the benchmark identity or evaluation target.
    """
    benchmark = row["benchmark"]
    task_text = ""
    context_text = ""
    option_text = ""

    if benchmark == "mmlu_medical":
        task_text = str(row.get("question", "")).strip()
        choices = parse_json_field(row, "choices_json", [])
        option_text = "\n".join(f"{chr(65 + i)}. {choice}" for i, choice in enumerate(choices))
    elif benchmark == "healthbench":
        messages = parse_json_field(row, "messages_json", [])
        task_text = "\n".join(
            str(message.get("content", "")).strip()
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        )
    elif benchmark == "medagentbench":
        task_text = str(row.get("instruction", "")).strip()
        context_text = str(row.get("context", "")).strip()

    sections = [
        ("Task", task_text),
        ("Context", context_text),
        ("Answer options", option_text),
    ]
    return "\n\n".join(f"{title}:\n{content}" for title, content in sections if content)


def compact_router_text(row: dict[str, Any], input_mode: str = "metadata_rich") -> str:
    if input_mode == "metadata_rich":
        return metadata_rich_router_text(row)
    if input_mode == "content_only":
        return content_only_router_text(row)
    raise ValueError(f"Unsupported input mode: {input_mode}")


def convert_row(row: dict[str, Any], input_mode: str = "metadata_rich") -> dict[str, Any]:
    benchmark = row["benchmark"]
    model_label = MODEL_BY_BENCHMARK[benchmark]
    tool_label = TOOL_BY_BENCHMARK[benchmark]
    prompt_label = PROMPT_BY_BENCHMARK[benchmark]
    return {
        "id": row["id"],
        "source_id": row["source_id"],
        "benchmark": benchmark,
        "text": compact_router_text(row, input_mode=input_mode),
        "model_label": model_label,
        "tool_label": tool_label,
        "prompt_label": prompt_label,
        "model_label_id": MODEL_LABELS.index(model_label),
        "tool_label_id": TOOL_LABELS.index(tool_label),
        "prompt_label_id": PROMPT_LABELS.index(prompt_label),
        "router_metadata_json": row.get("router_metadata_json") or "{}",
    }


def split_dataset(ds: Dataset, seed: int) -> DatasetDict:
    first = ds.train_test_split(test_size=0.2, seed=seed, stratify_by_column="benchmark")
    second = first["test"].train_test_split(test_size=0.5, seed=seed, stratify_by_column="benchmark")
    return DatasetDict(train=first["train"], validation=second["train"], test=second["test"])


def source_split_dataset(source_name: str, input_mode: str) -> DatasetDict:
    if Path(source_name).exists():
        source_ds = load_from_disk(source_name)
        splits = {}
        for split_name in ("train", "validation", "test"):
            rows = [
                convert_row(dict(row), input_mode=input_mode)
                for row in source_ds[split_name]
                if row["benchmark"] in MODEL_BY_BENCHMARK
            ]
            splits[split_name] = Dataset.from_list(rows).class_encode_column("benchmark")
        return DatasetDict(splits)

    splits = {}
    for split_name in ("train", "validation", "test"):
        source = load_dataset(source_name, split=split_name)
        rows = [convert_row(dict(row), input_mode=input_mode) for row in source if row["benchmark"] in MODEL_BY_BENCHMARK]
        splits[split_name] = Dataset.from_list(rows).class_encode_column("benchmark")
    return DatasetDict(splits)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--out-dir", default="hf_out/umsb-routing-classification")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--input-mode",
        choices=["metadata_rich", "content_only"],
        default="metadata_rich",
        help="Router text representation. content_only removes explicit benchmark/scenario metadata.",
    )
    parser.add_argument(
        "--use-source-splits",
        action="store_true",
        help="Use source train/validation/test splits directly instead of splitting one source split.",
    )
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    if args.use_source_splits:
        split = source_split_dataset(args.source, args.input_mode)
    else:
        source = load_dataset(args.source, split=args.source_split)
        rows = [convert_row(dict(row), input_mode=args.input_mode) for row in source if row["benchmark"] in MODEL_BY_BENCHMARK]
        ds = Dataset.from_list(rows).class_encode_column("benchmark")
        split = split_dataset(ds, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split.save_to_disk(str(out_dir))
    metadata = {
        "source": args.source,
        "source_split": args.source_split,
        "input_mode": args.input_mode,
        "use_source_splits": args.use_source_splits,
        "model_labels": MODEL_LABELS,
        "tool_labels": TOOL_LABELS,
        "prompt_labels": PROMPT_LABELS,
        "counts": {name: Counter(split[name]["benchmark"]).most_common() for name in split},
        "tool_counts": {name: Counter(split[name]["tool_label"]).most_common() for name in split},
        "prompt_counts": {name: Counter(split[name]["prompt_label"]).most_common() for name in split},
    }
    (out_dir / "routing_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))

    if args.push:
        split.push_to_hub(args.repo_id)
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
