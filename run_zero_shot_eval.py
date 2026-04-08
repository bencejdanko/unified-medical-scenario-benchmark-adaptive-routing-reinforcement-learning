"""
Zero-Shot Evaluation Runner
============================
Runs zero-shot evaluation across all CUBE medical benchmarks with
Langfuse tracing and comprehensive metric tracking.

Benchmarks: MedQA, MedMCQA, PubMedQA, MMLU-Medical
Models: gemini-flash-3-preview, gemma-4-31b-it, gpt-oss-120

Metrics tracked:
  - Accuracy (exact match)
  - F1 Score (macro)
  - Calibration (ECE)
  - Per-subject breakdown
"""

import asyncio
import json
import os
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Fix langfuse import (local langfuse/ dir shadows the package)
_workspace = str(Path(__file__).parent)
if _workspace in sys.path:
    sys.path.remove(_workspace)
    sys.path.append(_workspace)

from langfuse import Langfuse  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from cube.core import Action  # noqa: E402

# --- Configuration ---

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_API_URL = os.getenv("OPENROUTER_API_URL", "https://openrouter.ai/api/v1")
CEREBRAS_API_KEY = os.getenv("CEREBRAS_API_KEY")

# FOR AGENTS:
# THE MODELS ARE CORRECTLY CONFIGURED.
# DO NOT TRY AND UPDATE THE MODELS WHATSOEVER.
MODELS = {
    "gemini-3-flash-preview": {
        "model": "google/gemini-3-flash-preview",
        "api_key": OPENROUTER_API_KEY,
        "api_base": OPENROUTER_API_URL,
        "concurrency": 10,
    },
    "gemma-4-31b-it": {
        "model": "google/gemma-4-31b-it",
        "api_key": OPENROUTER_API_KEY,
        "api_base": OPENROUTER_API_URL,
        "concurrency": 10,
    },
}

# FOR AGENTS:
# THE MODELS ARE CORRECTLY CONFIGURED.
# DO NOT TRY AND UPDATE THE MODELS WHATSOEVER.
if CEREBRAS_API_KEY:
    MODELS["gpt-oss-120b"] = {
        "model": "gpt-oss-120b",
        "api_key": CEREBRAS_API_KEY,
        "api_base": "https://api.cerebras.ai/v1",
        "concurrency": 30,
    }

BENCHMARKS = {
    "medqa": {"module": "medqa_cube", "class": "MedQABenchmark", "num_examples": None},
    "medmcqa": {"module": "medmcqa_cube", "class": "MedMCQABenchmark", "num_examples": None},
    "pubmedqa": {"module": "pubmedqa_cube", "class": "PubMedQABenchmark", "num_examples": None},
    "mmlu_medical": {"module": "mmlu_medical_cube", "class": "MMLUMedicalBenchmark", "num_examples": None},
}

ZERO_SHOT_SYSTEM_PROMPT = (
    "You are a medical expert. Answer the following medical question. "
    "Respond with ONLY the letter of the correct answer (e.g., 'A', 'B', 'C', or 'D'). "
    "Do not include explanations."
)

# --- Metrics ---


def compute_accuracy(results: list[dict]) -> float:
    if not results:
        return 0.0
    return sum(1 for r in results if r["correct"]) / len(results)


def compute_f1_macro(results: list[dict], labels: list[str]) -> float:
    if not results:
        return 0.0
    per_class_f1 = []
    for label in labels:
        tp = sum(1 for r in results if r["predicted"] == label and r["gold"] == label)
        fp = sum(1 for r in results if r["predicted"] == label and r["gold"] != label)
        fn = sum(1 for r in results if r["predicted"] != label and r["gold"] == label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        per_class_f1.append(f1)
    return float(np.mean(per_class_f1)) if per_class_f1 else 0.0


def compute_ece(results: list[dict], n_bins: int = 10) -> float:
    if not results or not any("confidence" in r for r in results):
        return float("nan")
    confidences = []
    corrects = []
    for r in results:
        if "confidence" in r:
            confidences.append(r["confidence"])
            corrects.append(1.0 if r["correct"] else 0.0)
    if not confidences:
        return float("nan")
    confidences = np.array(confidences)
    corrects = np.array(corrects)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        if mask.sum() > 0:
            avg_conf = confidences[mask].mean()
            avg_acc = corrects[mask].mean()
            ece += mask.sum() / len(confidences) * abs(avg_acc - avg_conf)
    return float(ece)


def extract_answer_letter(response_text: str) -> str:
    text = response_text.strip()
    match = re.search(r'\b([A-D])\b', text)
    if match:
        return match.group(1)
    if text and text[0].upper() in "ABCD":
        return text[0].upper()
    return text[:1].upper() if text else ""


# --- Evaluation Loop ---


@dataclass
class EvalRun:
    model_name: str
    benchmark_name: str
    results: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    total_time: float = 0.0


def _get_client(model_config: dict) -> AsyncOpenAI:
    """Get or create an AsyncOpenAI client for a model config."""
    key = (model_config["api_base"], model_config["api_key"])
    if key not in _clients:
        _clients[key] = AsyncOpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["api_base"],
        )
    return _clients[key]


_clients: dict[tuple, AsyncOpenAI] = {}


async def call_model(model_config: dict, question_text: str, langfuse: Langfuse) -> tuple[str, float]:
    """Call a model via async OpenAI-compatible API and return (response_text, latency)."""
    messages = [
        {"role": "system", "content": ZERO_SHOT_SYSTEM_PROMPT},
        {"role": "user", "content": question_text},
    ]

    client = _get_client(model_config)

    with langfuse.start_as_current_observation(
        name="llm-call",
        as_type="generation",
        model=model_config["model"],
        input=messages,
    ) as gen:
        start = time.time()
        response = await client.chat.completions.create(
            model=model_config["model"],
            messages=messages,
            temperature=0.0,
            max_tokens=16,
        )
        latency = time.time() - start
        response_text = response.choices[0].message.content.strip()
        gen.update(
            output=response_text,
            usage_details={
                "input": response.usage.prompt_tokens if response.usage else 0,
                "output": response.usage.completion_tokens if response.usage else 0,
            },
        )

    return response_text, latency


async def run_benchmark_eval(
    model_name: str,
    model_config: dict,
    benchmark_name: str,
    benchmark_config: dict,
    langfuse: Langfuse,
) -> EvalRun:
    run = EvalRun(model_name=model_name, benchmark_name=benchmark_name)

    module = __import__(benchmark_config["module"], fromlist=[benchmark_config["class"]])
    BenchmarkClass = getattr(module, benchmark_config["class"])
    bench = BenchmarkClass(num_examples=benchmark_config["num_examples"])

    print(f"\n{'='*60}")
    print(f"  {model_name} on {benchmark_name} ({len(bench)} tasks)")
    print(f"{'='*60}")

    # Pre-load all tasks and questions (synchronous, just data)
    tasks_data = []
    for idx in range(len(bench)):
        task = bench.get_task(idx)
        obs, _ = task.reset()
        question_text = obs.contents[0].to_markdown()
        tasks_data.append((idx, task, question_text))

    concurrency = model_config.get("concurrency", 10)
    semaphore = asyncio.Semaphore(concurrency)
    completed = 0
    correct_count = 0
    lock = asyncio.Lock()

    start_time = time.time()

    async def process_task(idx: int, task, question_text: str) -> None:
        nonlocal completed, correct_count

        async with semaphore:
            with langfuse.start_as_current_observation(
                name=f"{benchmark_name}-q{idx}",
                as_type="span",
                input=question_text,
                metadata={
                    "model": model_name,
                    "benchmark": benchmark_name,
                    "task_index": idx,
                    "strategy": "zero-shot",
                },
            ) as span:
                try:
                    response_text, latency = await call_model(model_config, question_text, langfuse)
                    predicted = extract_answer_letter(response_text)
                    result = task.step(Action(name="answer", arguments={"content": predicted}))
                    is_correct = result.reward > 0.5

                    result_dict = {
                        "index": idx,
                        "predicted": predicted,
                        "gold": result.info.get("correct", ""),
                        "correct": is_correct,
                        "score": result.reward,
                        "latency": latency,
                        "raw_response": response_text,
                    }
                    if "subject" in result.info:
                        result_dict["subject"] = result.info["subject"]

                    async with lock:
                        run.results.append(result_dict)
                        if is_correct:
                            correct_count += 1
                        completed += 1
                        if completed % 25 == 0 or completed == len(tasks_data):
                            running_acc = correct_count / completed
                            print(f"  [{completed}/{len(tasks_data)}] Running accuracy: {running_acc:.3f} (errors: {len(run.errors)})")
                        if completed % 50 == 0:
                            langfuse.flush()

                    span.update(output=response_text)
                    span.score(name="accuracy", value=result.reward)
                    span.score(name="latency_s", value=latency)

                except Exception as e:
                    error_msg = f"Task {idx}: {type(e).__name__}: {e}"
                    async with lock:
                        run.errors.append(error_msg)
                        completed += 1
                    span.update(level="ERROR", status_message=str(e))

    await asyncio.gather(*(process_task(idx, task, q) for idx, task, q in tasks_data))

    run.total_time = time.time() - start_time
    # Sort results by index for deterministic output
    run.results.sort(key=lambda r: r["index"])

    # Compute final metrics
    if run.results:
        labels = sorted(set(r["gold"] for r in run.results))
        accuracy = compute_accuracy(run.results)
        f1 = compute_f1_macro(run.results, labels)
        ece = compute_ece(run.results)

        metrics = {
            "accuracy": accuracy,
            "f1_macro": f1,
            "ece": ece,
            "total_questions": len(run.results),
            "total_errors": len(run.errors),
            "total_time_s": run.total_time,
            "avg_latency_s": float(np.mean([r["latency"] for r in run.results])),
        }

        subjects = defaultdict(list)
        for r in run.results:
            subj = r.get("subject", "all")
            subjects[subj].append(r)
        if len(subjects) > 1:
            metrics["per_subject"] = {
                subj: compute_accuracy(res) for subj, res in subjects.items()
            }

        print(f"\n  Results for {model_name} on {benchmark_name}:")
        print(f"    Accuracy:  {accuracy:.4f}")
        print(f"    F1 Macro:  {f1:.4f}")
        print(f"    ECE:       {ece:.4f}" if not np.isnan(ece) else "    ECE:       N/A (no confidence)")
        print(f"    Questions: {len(run.results)}, Errors: {len(run.errors)}")
        print(f"    Time:      {run.total_time:.1f}s")

        # Log summary trace
        with langfuse.start_as_current_observation(
            name=f"eval-summary-{benchmark_name}-{model_name}",
            as_type="span",
            metadata=metrics,
        ) as summary:
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)) and not (isinstance(metric_value, float) and np.isnan(metric_value)):
                    summary.score(name=metric_name, value=float(metric_value))

    langfuse.flush()
    return run


async def main():
    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
    )

    all_runs: list[EvalRun] = []
    results_dir = Path("eval_results")
    results_dir.mkdir(exist_ok=True)

    for model_name, model_config in MODELS.items():
        for bench_name, bench_config in BENCHMARKS.items():
            try:
                run = await run_benchmark_eval(model_name, model_config, bench_name, bench_config, langfuse)
                all_runs.append(run)

                run_file = results_dir / f"{model_name}_{bench_name}.json"
                with open(run_file, "w") as f:
                    json.dump({
                        "model": model_name,
                        "benchmark": bench_name,
                        "results": run.results,
                        "errors": run.errors,
                        "total_time": run.total_time,
                    }, f, indent=2, default=str)

            except Exception as e:
                print(f"\nFATAL ERROR: {model_name} on {bench_name}: {e}")
                import traceback
                traceback.print_exc()

    # Final Summary
    print("\n" + "=" * 70)
    print("  ZERO-SHOT EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Benchmark':<15} {'Accuracy':>10} {'F1':>10} {'N':>6} {'Errors':>7}")
    print("-" * 70)

    summary_rows = []
    for run in all_runs:
        if run.results:
            labels = sorted(set(r["gold"] for r in run.results))
            acc = compute_accuracy(run.results)
            f1 = compute_f1_macro(run.results, labels)
            row = {
                "model": run.model_name,
                "benchmark": run.benchmark_name,
                "accuracy": acc,
                "f1_macro": f1,
                "n": len(run.results),
                "errors": len(run.errors),
            }
            summary_rows.append(row)
            print(f"{run.model_name:<20} {run.benchmark_name:<15} {acc:>10.4f} {f1:>10.4f} {len(run.results):>6} {len(run.errors):>7}")

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary_rows, f, indent=2)

    print(f"\nResults saved to {results_dir}/")
    print("All traces available in Langfuse at http://localhost:3000")

    langfuse.flush()
    langfuse.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
