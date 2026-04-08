"""
Modal: RAG Evaluation on Medical QA Benchmarks
================================================
Runs the full RAG eval pipeline on Modal cloud infrastructure:
  - Loads pre-built corpus + embeddings from Modal volume
  - Builds FAISS index in-memory (32GB RAM available)
  - Runs CUBE benchmarks with RAG-augmented prompts
  - Calls external LLM APIs (Cerebras gpt-oss-120b, OpenRouter)
  - Saves results to Modal volume and syncs back

Usage:
    modal run modal_rag_eval.py
    modal run modal_rag_eval.py --model gpt-oss-120b --benchmark medqa
    modal run modal_rag_eval.py --model gpt-oss-120b --top-k 5
    modal run modal_rag_eval.py --model gpt-oss-120b --benchmark "medqa,medmcqa,pubmedqa,mmlu_medical" --top-k 5
"""

import modal

VOLUME_NAME = "medqa-data-volume"
WORKSPACE = "/home/bence/medqa-cube-adaptive-curriculum-tool-calling"
CUBE_SRC = "/home/bence/cube-standard/src/cube"

volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "numpy>=1.26",
        "faiss-cpu>=1.7",
        "sentence-transformers>=3.0",
        "transformers>=4.40",
        "torch>=2.2",
        "openai>=1.30",
        "tqdm",
        "pydantic>=2.0",
        "pydantic-settings",
        "python-dotenv",
        "docstring-parser",
        "rich",
        "fastapi",
        "uvicorn",
        "pillow",
    )
    # Bake CUBE framework source into image
    .add_local_dir(CUBE_SRC, remote_path="/app/cube")
    # Bake benchmark modules into image
    .add_local_dir(f"{WORKSPACE}/medqa_cube", remote_path="/app/medqa_cube")
    .add_local_dir(f"{WORKSPACE}/medmcqa_cube", remote_path="/app/medmcqa_cube")
    .add_local_dir(f"{WORKSPACE}/pubmedqa_cube", remote_path="/app/pubmedqa_cube")
    .add_local_dir(f"{WORKSPACE}/mmlu_medical_cube", remote_path="/app/mmlu_medical_cube")
)

app = modal.App("medical-rag-eval", image=image)


# ---------------------------------------------------------------------------
# FAISS-based RAG retriever (runs in Modal with 32GB RAM)
# ---------------------------------------------------------------------------

class ModalRAG:
    """In-memory FAISS retrieval over medical corpus."""

    def __init__(self, chunks_path: str, embeddings_path: str, model_name: str = "NeuML/pubmedbert-base-embeddings"):
        import json
        import faiss

        print("[RAG] Loading chunks...")
        self.chunks = []
        with open(chunks_path) as f:
            for line in f:
                self.chunks.append(json.loads(line))
        print(f"[RAG] Loaded {len(self.chunks)} chunks")

        print("[RAG] Loading embeddings...")
        import numpy as np
        embeddings = np.load(embeddings_path).astype(np.float32)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]

        print(f"[RAG] Building FAISS index ({embeddings.shape[0]} vectors, dim={dim})...")
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        print(f"[RAG] Loading embedding model: {model_name}")
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(model_name, device="cpu")

        print("[RAG] Ready.")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        import numpy as np
        vec = self.embed_model.encode([query], normalize_embeddings=True).astype(np.float32)
        scores, indices = self.index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        return results

    def build_prompt(self, question_text: str, retrieved: list[dict]) -> str:
        if not retrieved:
            return question_text
        context_parts = []
        for i, chunk in enumerate(retrieved, 1):
            source = chunk.get("source", "unknown")
            title = chunk.get("title", "")
            header = f"[{source}] {title}" if title else f"[{source}]"
            context_parts.append(f"--- Reference {i} ({header}) ---\n{chunk['text']}")
        context_block = "\n\n".join(context_parts)
        return (
            f"Use the following medical references to help answer the question. "
            f"If the references are not relevant, rely on your own knowledge.\n\n"
            f"{context_block}\n\n"
            f"--- Question ---\n{question_text}"
        )


# ---------------------------------------------------------------------------
# Evaluation logic (adapted from run_rag_eval.py for Modal)
# ---------------------------------------------------------------------------

@app.function(
    volumes={"/data": volume},
    secrets=[
        modal.Secret.from_name("openrouter-secret"),
        modal.Secret.from_local_environ(["CEREBRAS_API_KEY", "OPENROUTER_API_KEY"]),
    ],
    timeout=7200,
    memory=32768,
    cpu=4,
)
def run_rag_eval(
    model_name: str = "gpt-oss-120b",
    benchmarks: list[str] = ["medqa", "medmcqa", "pubmedqa", "mmlu_medical"],
    top_k: int = 5,
):
    """Run RAG evaluation on Modal."""
    import asyncio
    import json
    import os
    import re
    import sys
    import time
    from collections import defaultdict
    from dataclasses import dataclass, field

    import numpy as np

    # Set up sys.path for cube + benchmark modules
    sys.path.insert(0, "/app")

    from cube.core import Action
    from openai import AsyncOpenAI

    # ---- Model config ----
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
    CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

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
    if CEREBRAS_API_KEY:
        MODELS["gpt-oss-120b"] = {
            "model": "gpt-oss-120b",
            "api_key": CEREBRAS_API_KEY,
            "api_base": "https://api.cerebras.ai/v1",
            "concurrency": 30,
            "max_tokens": 1024,
        }

    BENCHMARK_MAP = {
        "medqa": {"module": "medqa_cube", "class": "MedQABenchmark", "num_examples": None},
        "medmcqa": {"module": "medmcqa_cube", "class": "MedMCQABenchmark", "num_examples": None},
        "pubmedqa": {"module": "pubmedqa_cube", "class": "PubMedQABenchmark", "num_examples": None},
        "mmlu_medical": {"module": "mmlu_medical_cube", "class": "MMLUMedicalBenchmark", "num_examples": None},
    }

    RAG_SYSTEM_PROMPT = (
        "You are a medical expert. Answer the following medical question using the provided "
        "reference materials when relevant. If the references are not helpful, rely on your "
        "own medical knowledge. "
        'Respond with ONLY valid JSON in this exact format: {"answer": "X", "confidence": Y} '
        "where X is the letter of the correct answer (A, B, C, or D) and Y is your confidence "
        "as a decimal between 0.0 and 1.0. Do not include explanations or any other text."
    )

    # ---- Metrics ----
    def compute_accuracy(results):
        if not results:
            return 0.0
        return sum(1 for r in results if r["correct"]) / len(results)

    def compute_f1_macro(results, labels):
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

    def compute_ece(results, n_bins=10):
        if not results or not any("confidence" in r for r in results):
            return float("nan")
        confidences, corrects = [], []
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
                ece += mask.sum() / len(confidences) * abs(corrects[mask].mean() - confidences[mask].mean())
        return float(ece)

    def parse_model_response(response_text):
        text = response_text.strip()
        try:
            parsed = json.loads(text)
            answer = str(parsed.get("answer", "")).strip().upper()
            confidence = float(parsed.get("confidence", 0.5))
            confidence = max(0.0, min(1.0, confidence))
            if answer and answer[0] in "ABCD":
                return answer[0], confidence
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
        json_match = re.search(r'\{[^}]*"answer"\s*:\s*"([A-D])"[^}]*"confidence"\s*:\s*([\d.]+)[^}]*\}', text)
        if json_match:
            return json_match.group(1), max(0.0, min(1.0, float(json_match.group(2))))
        match = re.search(r'\b([A-D])\b', text)
        if match:
            return match.group(1), 0.5
        if text and text[0].upper() in "ABCD":
            return text[0].upper(), 0.5
        return (text[:1].upper() if text else ""), 0.5

    # ---- Initialize RAG ----
    print("=" * 60)
    print(f"  Modal RAG Evaluation")
    print(f"  Model: {model_name}, Benchmarks: {benchmarks}, top_k={top_k}")
    print("=" * 60)

    rag = ModalRAG(
        chunks_path="/data/rag_corpus/chunks.jsonl",
        embeddings_path="/data/rag_corpus/embeddings.npy",
    )

    if model_name not in MODELS:
        raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")

    model_config = MODELS[model_name]
    client = AsyncOpenAI(api_key=model_config["api_key"], base_url=model_config["api_base"])

    # ---- Normalize benchmark names (allow hyphens or underscores) ----
    benchmarks = [b.replace("-", "_") for b in benchmarks]
    valid = [b for b in benchmarks if b in BENCHMARK_MAP]
    invalid = [b for b in benchmarks if b not in BENCHMARK_MAP]
    if invalid:
        print(f"  WARNING: Unknown benchmarks skipped: {invalid}")
        print(f"  Available: {list(BENCHMARK_MAP.keys())}")
    if not valid:
        raise ValueError(f"No valid benchmarks specified. Available: {list(BENCHMARK_MAP.keys())}")
    benchmarks = valid
    print(f"  Will run {len(benchmarks)} benchmark(s): {benchmarks}")

    # ---- Run benchmarks ----
    all_summaries = []

    for bench_name in benchmarks:
        if bench_name not in BENCHMARK_MAP:
            print(f"  Skipping unknown benchmark: {bench_name}")
            continue

        bench_config = BENCHMARK_MAP[bench_name]
        module = __import__(bench_config["module"], fromlist=[bench_config["class"]])
        BenchmarkClass = getattr(module, bench_config["class"])
        bench = BenchmarkClass(num_examples=bench_config["num_examples"])

        print(f"\n{'='*60}")
        print(f"  [RAG top_k={top_k}] {model_name} on {bench_name} ({len(bench)} tasks)")
        print(f"{'='*60}")

        # Pre-load tasks and retrieve context
        print("  Retrieving context for all questions...")
        tasks_data = []
        for idx in range(len(bench)):
            task = bench.get_task(idx)
            obs, _ = task.reset()
            question_text = obs.contents[0].to_markdown()
            retrieved = rag.retrieve(question_text, top_k=top_k)
            augmented = rag.build_prompt(question_text, retrieved)
            tasks_data.append((idx, task, question_text, augmented, retrieved))
        print(f"  Context retrieved for {len(tasks_data)} questions")

        # Log one sample prompt for debugging
        if tasks_data:
            sample_idx, _, sample_q, sample_aug, sample_ret = tasks_data[0]
            print(f"\n  --- Sample Prompt (task #{sample_idx}, {bench_name}) ---")
            print(f"  System: {RAG_SYSTEM_PROMPT[:120]}...")
            print(f"  User prompt ({len(sample_aug)} chars, {len(sample_ret)} references):")
            # Print first 1500 chars of the augmented prompt
            preview = sample_aug[:1500]
            if len(sample_aug) > 1500:
                preview += f"\n  ... [truncated, {len(sample_aug) - 1500} chars remaining]"
            for line in preview.split("\n"):
                print(f"    {line}")
            print(f"  --- End Sample Prompt ---\n")

        # Async eval
        results = []
        errors = []

        async def run_all():
            nonlocal results, errors
            semaphore = asyncio.Semaphore(model_config.get("concurrency", 10))
            completed = 0
            correct_count = 0
            lock = asyncio.Lock()

            async def process(idx, task, question_text, augmented, retrieved):
                nonlocal completed, correct_count
                async with semaphore:
                    try:
                        messages = [
                            {"role": "system", "content": RAG_SYSTEM_PROMPT},
                            {"role": "user", "content": augmented},
                        ]
                        start = time.time()
                        response = await client.chat.completions.create(
                            model=model_config["model"],
                            messages=messages,
                            temperature=0.0,
                            max_tokens=model_config.get("max_tokens", 64),
                        )
                        latency = time.time() - start
                        content = response.choices[0].message.content
                        refusal = getattr(response.choices[0].message, "refusal", None)
                        response_text = (content or refusal or "").strip()

                        predicted, confidence = parse_model_response(response_text)
                        result = task.step(Action(name="answer", arguments={"content": predicted}))
                        is_correct = result.reward > 0.5

                        result_dict = {
                            "index": idx,
                            "predicted": predicted,
                            "confidence": confidence,
                            "gold": result.info.get("correct", ""),
                            "correct": is_correct,
                            "score": result.reward,
                            "latency": latency,
                            "raw_response": response_text,
                            "top_k": top_k,
                            "num_retrieved": len(retrieved),
                            "avg_retrieval_score": float(np.mean([r["score"] for r in retrieved])) if retrieved else 0.0,
                        }
                        if "subject" in result.info:
                            result_dict["subject"] = result.info["subject"]

                        async with lock:
                            results.append(result_dict)
                            if is_correct:
                                correct_count += 1
                            completed += 1
                            if completed % 25 == 0 or completed == len(tasks_data):
                                running_acc = correct_count / completed
                                print(f"  [{completed}/{len(tasks_data)}] Running acc: {running_acc:.3f} (errors: {len(errors)})")

                    except Exception as e:
                        error_msg = f"Task {idx}: {type(e).__name__}: {e}"
                        async with lock:
                            errors.append(error_msg)
                            completed += 1
                            if len(errors) <= 5:
                                print(f"  ⚠ {error_msg[:200]}")

            await asyncio.gather(*(
                process(idx, task, q, aug, ret) for idx, task, q, aug, ret in tasks_data
            ))

        start_time = time.time()
        asyncio.run(run_all())
        total_time = time.time() - start_time

        results.sort(key=lambda r: r["index"])

        # Metrics
        if results:
            labels = sorted(set(r["gold"] for r in results))
            accuracy = compute_accuracy(results)
            f1 = compute_f1_macro(results, labels)
            ece = compute_ece(results)

            summary = {
                "model": model_name,
                "benchmark": bench_name,
                "strategy": "rag-naive",
                "top_k": top_k,
                "accuracy": accuracy,
                "f1_macro": f1,
                "ece": ece,
                "n": len(results),
                "errors": len(errors),
                "total_time_s": total_time,
            }
            all_summaries.append(summary)

            ece_str = f"{ece:.4f}" if not np.isnan(ece) else "N/A"
            print(f"\n  Results for {model_name} on {bench_name} [RAG top_k={top_k}]:")
            print(f"    Accuracy:  {accuracy:.4f}")
            print(f"    F1 Macro:  {f1:.4f}")
            print(f"    ECE:       {ece_str}")
            print(f"    Questions: {len(results)}, Errors: {len(errors)}")
            print(f"    Time:      {total_time:.1f}s")

        # Save per-benchmark results
        out_dir = "/data/eval_results_rag"
        os.makedirs(out_dir, exist_ok=True)
        out_file = f"{out_dir}/{model_name}_{bench_name}_rag_k{top_k}.json"
        with open(out_file, "w") as f:
            json.dump({
                "model": model_name,
                "benchmark": bench_name,
                "strategy": "rag-naive",
                "top_k": top_k,
                "results": results,
                "errors": errors,
                "total_time": total_time,
            }, f, indent=2, default=str)

    # Save summary
    out_dir = "/data/eval_results_rag"
    with open(f"{out_dir}/summary_rag_{model_name}_k{top_k}.json", "w") as f:
        json.dump(all_summaries, f, indent=2)

    volume.commit()

    # Print final summary
    print("\n" + "=" * 70)
    print(f"  RAG EVALUATION SUMMARY (top_k={top_k})")
    print("=" * 70)
    print(f"{'Model':<20} {'Benchmark':<15} {'Accuracy':>10} {'F1':>10} {'ECE':>10} {'N':>6}")
    print("-" * 70)
    for s in all_summaries:
        ece_str = f"{s['ece']:.4f}" if not np.isnan(s['ece']) else "N/A"
        print(f"{s['model']:<20} {s['benchmark']:<15} {s['accuracy']:>10.4f} {s['f1_macro']:>10.4f} {ece_str:>10} {s['n']:>6}")

    print(f"\nResults saved to Modal volume: {out_dir}/")
    return all_summaries


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main(
    model: str = "gpt-oss-120b",
    benchmark: str = "medqa,medmcqa,pubmedqa,mmlu_medical",
    top_k: int = 5,
):
    benchmarks = [b.strip().replace("-", "_") for b in benchmark.split(",")]
    print(f"Running RAG eval on Modal: model={model}, benchmarks={benchmarks}, top_k={top_k}")
    print(f"  Total benchmarks to run: {len(benchmarks)}")
    summaries = run_rag_eval.remote(
        model_name=model,
        benchmarks=benchmarks,
        top_k=top_k,
    )
    print(f"\nDone! Downloading results from Modal volume...")

    # Download results locally
    import subprocess
    subprocess.run([
        "modal", "volume", "get", VOLUME_NAME,
        "eval_results_rag/", "eval_results_rag/",
    ], check=False)
    print("Results downloaded to eval_results_rag/")
