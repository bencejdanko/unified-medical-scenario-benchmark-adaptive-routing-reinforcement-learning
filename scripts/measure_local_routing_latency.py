from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any

from datasets import load_dataset, load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_unified_eval import DistilBertRouterRuntime, EpisodicMemoryRuntime


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * q)))
    return ordered[idx]


def summarize(values: list[float]) -> dict[str, float | None]:
    return {
        "n": len(values),
        "mean_ms": mean(values) * 1000 if values else None,
        "median_ms": median(values) * 1000 if values else None,
        "p95_ms": percentile(values, 0.95) * 1000 if values else None,
        "p99_ms": percentile(values, 0.99) * 1000 if values else None,
        "min_ms": min(values) * 1000 if values else None,
        "max_ms": max(values) * 1000 if values else None,
    }


def load_rows(dataset: str, split: str) -> list[dict[str, Any]]:
    if Path(dataset).exists():
        ds = load_from_disk(dataset)
        rows = ds[split]
    else:
        rows = load_dataset(dataset, split=split)
    return [dict(row) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure local router/memory overhead without OpenRouter calls.")
    parser.add_argument("--dataset", default="hf_out/unified_medical_scenario_benchmark")
    parser.add_argument("--split", default="test")
    parser.add_argument("--router-model-path", default="hf_out/umsb-distilbert-router")
    parser.add_argument("--episodic-memory-model-path", default=None)
    parser.add_argument("--episodic-memory-chunks-path", default=None)
    parser.add_argument("--episodic-memory-embeddings-path", default=None)
    parser.add_argument("--episodic-memory-top-k", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--output", default="eval_results/router_measure_analysis/local_routing_latency.json")
    args = parser.parse_args()

    rows = load_rows(args.dataset, args.split)
    router = DistilBertRouterRuntime(args.router_model_path)
    memory = None
    if args.episodic_memory_model_path and args.episodic_memory_chunks_path and args.episodic_memory_embeddings_path:
        memory = EpisodicMemoryRuntime(
            args.episodic_memory_model_path,
            args.episodic_memory_top_k,
            chunks_path=args.episodic_memory_chunks_path,
            embeddings_path=args.episodic_memory_embeddings_path,
        )

    for row in rows[: args.warmup]:
        router.route(row)
        if memory is not None:
            memory.retrieve(row)

    router_times: list[float] = []
    memory_times: list[float] = []
    combined_times: list[float] = []
    by_benchmark_router: dict[str, list[float]] = defaultdict(list)
    by_benchmark_memory: dict[str, list[float]] = defaultdict(list)
    by_benchmark_combined: dict[str, list[float]] = defaultdict(list)

    for row in rows:
        benchmark = str(row.get("benchmark", "unknown"))
        combined_start = time.perf_counter()
        start = time.perf_counter()
        router.route(row)
        router_elapsed = time.perf_counter() - start
        memory_elapsed = 0.0
        if memory is not None:
            start = time.perf_counter()
            memory.retrieve(row)
            memory_elapsed = time.perf_counter() - start
            memory_times.append(memory_elapsed)
            by_benchmark_memory[benchmark].append(memory_elapsed)
        combined_elapsed = time.perf_counter() - combined_start

        router_times.append(router_elapsed)
        combined_times.append(combined_elapsed)
        by_benchmark_router[benchmark].append(router_elapsed)
        by_benchmark_combined[benchmark].append(combined_elapsed)

    report = {
        "dataset": args.dataset,
        "split": args.split,
        "n": len(rows),
        "router_model_path": args.router_model_path,
        "episodic_memory_enabled": memory is not None,
        "episodic_memory_model_path": args.episodic_memory_model_path,
        "router_latency": summarize(router_times),
        "memory_latency": summarize(memory_times) if memory is not None else None,
        "combined_router_memory_latency": summarize(combined_times),
        "by_benchmark": {
            benchmark: {
                "router_latency": summarize(by_benchmark_router[benchmark]),
                "memory_latency": summarize(by_benchmark_memory[benchmark]) if memory is not None else None,
                "combined_router_memory_latency": summarize(by_benchmark_combined[benchmark]),
            }
            for benchmark in sorted(by_benchmark_router)
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
