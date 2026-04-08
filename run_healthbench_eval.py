"""
HealthBench Evaluation Runner
==============================
Runs HealthBench evaluation across baseline models with rubric-based
grading using gemini-3-flash-preview as the judge model.

Models: gemini-3-flash-preview, gemma-4-31b-it, gpt-oss-120b
Judge:  gemini-3-flash-preview (via OpenRouter)

Metrics tracked (per HealthBench):
  - Overall score (rubric-based, clipped to [0,1])
  - Per-tag scores (example-level and rubric-level)
  - Bootstrap standard deviations
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Fix langfuse import (local langfuse/ dir shadows the package)
_workspace = str(Path(__file__).parent)
if _workspace in sys.path:
    sys.path.remove(_workspace)
    sys.path.append(_workspace)

from langfuse import Langfuse  # noqa: E402
from openai import OpenAI  # noqa: E402

import healthbench_cube.healthbench_eval as _hb_eval_mod  # noqa: E402

# Redirect HealthBenchEval to use local data files instead of Azure blob URLs
_LOCAL_DATA_DIR = Path(__file__).parent / "healthbench_cube" / "data"
_hb_eval_mod.INPUT_PATH = str(_LOCAL_DATA_DIR / "healthbench.jsonl")
# Hard/consensus subsets: use remote URLs as fallback (only if files exist locally)
_hard_path = _LOCAL_DATA_DIR / "healthbench_hard.jsonl"
_consensus_path = _LOCAL_DATA_DIR / "healthbench_consensus.jsonl"
if _hard_path.exists():
    _hb_eval_mod.INPUT_PATH_HARD = str(_hard_path)
if _consensus_path.exists():
    _hb_eval_mod.INPUT_PATH_CONSENSUS = str(_consensus_path)

from healthbench_cube.healthbench_eval import HealthBenchEval  # noqa: E402
from healthbench_cube.types import MessageList, SamplerBase, SamplerResponse  # noqa: E402

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
    },
    "gemma-4-31b-it": {
        "model": "google/gemma-4-31b-it",
        "api_key": OPENROUTER_API_KEY,
        "api_base": OPENROUTER_API_URL,
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
    }

# Judge model for rubric grading
GRADER_CONFIG = {
    "model": "google/gemini-3-flash-preview",
    "api_key": OPENROUTER_API_KEY,
    "api_base": OPENROUTER_API_URL,
}


# --- Samplers ---


class OpenRouterSampler(SamplerBase):
    """Sampler that routes to OpenRouter or Cerebras-compatible APIs."""

    def __init__(self, model_config: dict, temperature: float = 0.5, max_tokens: int = 4096):
        self.client = OpenAI(
            api_key=model_config["api_key"],
            base_url=model_config["api_base"],
        )
        self.model = model_config["model"]
        self.temperature = temperature
        self.max_tokens = max_tokens

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2 ** trial
                print(f"  Retry {trial} after {exception_backoff}s: {e}")
                time.sleep(exception_backoff)
                trial += 1
                if trial > 8:
                    return SamplerResponse(
                        response_text="No response (max retries exceeded).",
                        response_metadata={"usage": None},
                        actual_queried_message_list=message_list,
                    )


# --- CLI ---


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HealthBench evaluation runner")
    parser.add_argument(
        "--model", "-m",
        type=str,
        nargs="+",
        choices=list(MODELS.keys()),
        default=None,
        help="Model(s) to evaluate (default: all). Choices: " + ", ".join(MODELS.keys()),
    )
    parser.add_argument(
        "--num-examples", "-n",
        type=int,
        default=None,
        help="Number of examples to evaluate (default: all)",
    )
    parser.add_argument(
        "--n-threads",
        type=int,
        default=4,
        help="Number of threads for parallel grading (default: 4)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["hard", "consensus"],
        default=None,
        help="HealthBench subset to evaluate (default: full)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    langfuse = Langfuse(
        secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
        host=os.getenv("LANGFUSE_BASE_URL", "http://localhost:3000"),
    )

    results_dir = Path("eval_results")
    results_dir.mkdir(exist_ok=True)

    models_to_run = {k: v for k, v in MODELS.items() if args.model is None or k in args.model}

    if not models_to_run:
        print(f"No matching models found. Available: {', '.join(MODELS.keys())}")
        return

    subset_label = args.subset or "full"
    print(f"HealthBench Evaluation ({subset_label})")
    print(f"Models: {', '.join(models_to_run.keys())}")
    print(f"Judge:  {GRADER_CONFIG['model']}")
    if args.num_examples:
        print(f"Examples: {args.num_examples}")
    print()

    # Build grader sampler (gemini-3-flash-preview)
    grader_sampler = OpenRouterSampler(GRADER_CONFIG, temperature=0.0, max_tokens=2048)

    summary_rows = []

    for model_name, model_config in models_to_run.items():
        print(f"\n{'=' * 60}")
        print(f"  Evaluating {model_name} on HealthBench ({subset_label})")
        print(f"{'=' * 60}")

        # Build the sampler for the model under test
        model_sampler = OpenRouterSampler(model_config, temperature=0.5, max_tokens=4096)

        # Build the HealthBench eval
        eval_instance = HealthBenchEval(
            grader_model=grader_sampler,
            num_examples=args.num_examples,
            n_threads=args.n_threads,
            subset_name=args.subset,
        )

        # Log to Langfuse
        with langfuse.start_as_current_observation(
            name=f"healthbench-{subset_label}-{model_name}",
            as_type="span",
            metadata={
                "model": model_name,
                "benchmark": f"healthbench_{subset_label}",
                "judge": GRADER_CONFIG["model"],
                "num_examples": args.num_examples,
                "strategy": "zero-shot",
            },
        ) as span:
            start_time = time.time()
            result = eval_instance(model_sampler)
            total_time = time.time() - start_time

            # Extract metrics
            metrics = result.metrics or {}
            overall_score = result.score

            print(f"\n  Results for {model_name}:")
            print(f"    Overall Score: {overall_score:.4f}" if overall_score is not None else "    Overall Score: N/A")
            for k, v in sorted(metrics.items()):
                if isinstance(v, float):
                    print(f"    {k}: {v:.4f}")
            print(f"    Time: {total_time:.1f}s")

            # Log scores to Langfuse
            if overall_score is not None:
                span.score(name="overall_score", value=float(overall_score))
            for metric_name, metric_value in metrics.items():
                if isinstance(metric_value, (int, float)) and not (isinstance(metric_value, float) and np.isnan(metric_value)):
                    span.score(name=metric_name, value=float(metric_value))

        # Save per-model results
        run_file = results_dir / f"{model_name}_healthbench_{subset_label}.json"
        run_data = {
            "model": model_name,
            "benchmark": f"healthbench_{subset_label}",
            "judge_model": GRADER_CONFIG["model"],
            "overall_score": overall_score,
            "metrics": {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics.items()},
            "total_time_s": total_time,
            "num_examples": len(result.htmls),
            "metadata": result.metadata,
        }
        with open(run_file, "w") as f:
            json.dump(run_data, f, indent=2, default=str)
        print(f"  Saved: {run_file}")

        # Save HTML report
        from healthbench_cube.common import make_report
        report_file = results_dir / f"{model_name}_healthbench_{subset_label}.html"
        report_file.write_text(make_report(result))
        print(f"  Report: {report_file}")

        summary_rows.append({
            "model": model_name,
            "benchmark": f"healthbench_{subset_label}",
            "overall_score": overall_score,
            "num_examples": len(result.htmls),
            "total_time_s": total_time,
        })

        langfuse.flush()

    # Final Summary
    print("\n" + "=" * 70)
    print("  HEALTHBENCH EVALUATION SUMMARY")
    print("=" * 70)
    print(f"{'Model':<25} {'Score':>10} {'N':>6} {'Time':>8}")
    print("-" * 55)
    for row in summary_rows:
        score_str = f"{row['overall_score']:.4f}" if row["overall_score"] is not None else "N/A"
        print(f"{row['model']:<25} {score_str:>10} {row['num_examples']:>6} {row['total_time_s']:>7.1f}s")

    # Save summary
    summary_file = results_dir / f"healthbench_{subset_label}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_rows, f, indent=2, default=str)
    print(f"\nResults saved to {results_dir}/")
    print("All traces available in Langfuse at http://localhost:3000")

    langfuse.flush()
    langfuse.shutdown()


if __name__ == "__main__":
    main()
