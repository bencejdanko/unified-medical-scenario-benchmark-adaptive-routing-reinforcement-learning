"""Create and optionally upload a recursive-character episodic memory dataset.

This consumes unified eval JSONL outputs, prefers `*_episodic_memory_candidates`
files, and writes chunks suitable for training/retrieval with
sentence-transformers/all-mpnet-base-v2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict


DEFAULT_INPUT_DIR = "eval_results/unified_gemini3flash_train_"
DEFAULT_REPO = "bdanko/umsb-episodic-memory"


def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    separators = ["\n\n", "\n", ". ", " ", ""]

    def split_with(seps: list[str], value: str) -> list[str]:
        if len(value) <= chunk_size:
            return [value]
        sep = seps[0]
        if sep == "":
            return [value[i : i + chunk_size] for i in range(0, len(value), chunk_size - chunk_overlap)]
        pieces = value.split(sep)
        if len(pieces) == 1:
            return split_with(seps[1:], value)
        chunks: list[str] = []
        current = ""
        for piece in pieces:
            candidate = piece if not current else current + sep + piece
            if len(candidate) <= chunk_size:
                current = candidate
                continue
            if current:
                chunks.extend(split_with(seps[1:], current))
            current = piece
        if current:
            chunks.extend(split_with(seps[1:], current))
        return chunks

    raw_chunks = [chunk.strip() for chunk in split_with(separators, text) if chunk.strip()]
    if chunk_overlap <= 0 or len(raw_chunks) <= 1:
        return raw_chunks
    overlapped = []
    previous_tail = ""
    for chunk in raw_chunks:
        combined = (previous_tail + "\n" + chunk).strip() if previous_tail else chunk
        overlapped.append(combined[: chunk_size + chunk_overlap])
        previous_tail = chunk[-chunk_overlap:]
    return overlapped


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def failure_files(input_dir: Path) -> list[Path]:
    files = sorted(input_dir.glob("*_episodic_memory_candidates.jsonl"))
    if files:
        return files
    return sorted(input_dir.glob("*_episodes.jsonl"))


def first_user_text(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "user":
            return str(message.get("content", ""))
    return ""


def compact_transcript(transcript: list[dict[str, Any]], max_turns: int = 8) -> str:
    lines = []
    for message in transcript[:max_turns]:
        role = message.get("role", "unknown")
        content = str(message.get("content", "")).strip()
        if len(content) > 900:
            content = content[:900] + "..."
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def memory_text(row: dict[str, Any]) -> str:
    benchmark = row.get("benchmark", "")
    error = row.get("error")
    reward_details = row.get("reward_details") if isinstance(row.get("reward_details"), dict) else {}
    prediction = row.get("prediction") if isinstance(row.get("prediction"), dict) else {}
    general_advice = {
        "mmlu_medical": (
            "Avoid over-explaining multiple-choice answers. Parse all options carefully, "
            "then return the single best answer in the required JSON format."
        ),
        "healthbench": (
            "Address the user's clinical concern directly, include safety-relevant caveats, "
            "and cover all rubric dimensions instead of giving a narrow or generic response."
        ),
        "medagentbench": (
            "Use only the permitted FHIR action format, inspect tool responses before finishing, "
            "and call FINISH only with the requested final values."
        ),
    }.get(benchmark, "Follow the task format exactly and verify the final answer against the request.")
    parts = [
        f"benchmark: {benchmark}",
        f"task_id: {row.get('task_id', '')}",
        f"reward: {row.get('reward')}",
        f"failure_mode: {error or reward_details.get('status') or 'zero_reward'}",
        f"general_advice: {general_advice}",
        f"task_request: {first_user_text(row.get('prompt_messages') or row.get('transcript') or [])}",
        f"model_prediction_summary: {json.dumps(prediction, ensure_ascii=False, default=str)[:1200]}",
        f"judge_or_scorer_feedback: {json.dumps(reward_details, ensure_ascii=False, default=str)[:1600]}",
        f"traversal_excerpt: {compact_transcript(row.get('transcript') or [])}",
    ]
    return "\n".join(part for part in parts if part.strip())


def build_dataset(input_dir: Path, chunk_size: int, chunk_overlap: int) -> DatasetDict:
    chunks = []
    for path in failure_files(input_dir):
        for row in read_jsonl(path):
            reward = row.get("reward")
            if reward not in (0, 0.0, None) and not row.get("episodic_memory_candidate"):
                continue
            text = memory_text(row)
            for idx, chunk in enumerate(recursive_split(text, chunk_size, chunk_overlap)):
                chunks.append(
                    {
                        "id": f"{row.get('task_id', 'unknown')}:{idx}",
                        "task_id": row.get("task_id", ""),
                        "source_id": row.get("source_id", ""),
                        "benchmark": row.get("benchmark", ""),
                        "model": row.get("model", ""),
                        "reward": row.get("reward"),
                        "chunk_index": idx,
                        "text": chunk,
                    }
                )
    ds = Dataset.from_list(chunks)
    if len(ds) == 0:
        return DatasetDict(train=ds)
    split = ds.train_test_split(test_size=min(0.1, max(1 / len(ds), 0.01)), seed=17)
    return DatasetDict(train=split["train"], validation=split["test"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default=DEFAULT_INPUT_DIR)
    parser.add_argument("--repo-id", default=DEFAULT_REPO)
    parser.add_argument("--out-dir", default="hf_out/umsb-episodic-memory")
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--chunk-overlap", type=int, default=120)
    parser.add_argument("--push", action="store_true")
    args = parser.parse_args()

    ds = build_dataset(Path(args.input_dir), args.chunk_size, args.chunk_overlap)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    metadata = {
        "input_dir": args.input_dir,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "splits": {name: len(ds[name]) for name in ds},
    }
    (out_dir / "episodic_memory_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(json.dumps(metadata, indent=2))
    if args.push:
        ds.push_to_hub(args.repo_id)
        print(f"Pushed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
