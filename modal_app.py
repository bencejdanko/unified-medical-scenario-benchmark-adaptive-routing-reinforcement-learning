"""
MEDQA RESEARCH SUITE - EVALUATION FRAMEWORK
"""

import modal
import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional, Any

# --- Configuration & Styling ---
DATA_PATH = Path("/data")
RESULTS_PATH = DATA_PATH / "results"

# Set up logging for cleaner output in production
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Image Definitions ---
# Lean image for API calls and data processing
base_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "openai",
        "datasets",
        "tqdm",
        "scikit-learn",
        "mcp",
        "httpx",
        "cube-standard"
    )
    .add_local_dir(".", remote_path="/root")
)

# Heavy image for local model inference (if needed later)
gpu_image = (
    base_image
    .pip_install(
        "torch",
        "transformers",
        "accelerate"
        # "vllm" # Uncomment if local vLLM is required
    )
)

app = modal.App("medqa-research-suite")
volume = modal.Volume.from_name("medqa-data-volume", create_if_missing=True)

# --- Prompting Strategies ---

class PromptStrategy:
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    COT = "cot"
    COT_FEW_SHOT = "cot_few_shot"
    TOT = "tree_of_thoughts"

class PromptManager:
    """Manages prompt construction for various research strategies."""
    
    @staticmethod
    def construct_prompt(strategy: str, question: str, options: List[str], few_shot_examples: Optional[List[Dict]] = None) -> str:
        base_system = "<|im_start|>system\nYou are a medical expert assistant. "
        
        if strategy == PromptStrategy.ZERO_SHOT:
            return (f"{base_system}Answer the multiple choice question by providing only the index of the correct option (0, 1, 2, or 3).<|im_end|>\n"
                    f"<|im_start|>user\nQuestion: {question}\nOptions: {options}\nAnswer:<|im_end|>\n<|im_start|>assistant\n")
            
        elif strategy == PromptStrategy.FEW_SHOT:
            prompt = f"{base_system}Here are several examples of medical questions and their correct answers. Based on these, answer the final question with only the index.<|im_end|>\n"
            if few_shot_examples:
                for ex in few_shot_examples:
                    prompt += f"<|im_start|>user\nQuestion: {ex['question']}\nOptions: {ex['options']}\nAnswer:<|im_end|>\n"
                    prompt += f"<|im_start|>assistant\n{ex['answer_idx']}<|im_end|>\n"
            prompt += f"<|im_start|>user\nQuestion: {question}\nOptions: {options}\nAnswer:<|im_end|>\n<|im_start|>assistant\n"
            return prompt

        elif strategy == PromptStrategy.COT:
            return (f"{base_system}Think step-by-step through the clinical reasoning process, then provide the final answer index (0, 1, 2, or 3) at the very end as 'Final Answer: [INDEX]'.<|im_end|>\n"
                    f"<|im_start|>user\nQuestion: {question}\nOptions: {options}\nLet's think step by step:<|im_end|>\n<|im_start|>assistant\n")

        elif strategy == PromptStrategy.COT_FEW_SHOT:
            prompt = f"{base_system}Think step-by-step. Use the examples below to understand the expected reasoning depth.<|im_end|>\n"
            if few_shot_examples:
                for ex in few_shot_examples:
                    prompt += f"<|im_start|>user\nQuestion: {ex['question']}\nOptions: {ex['options']}\nAnswer:<|im_end|>\n"
                    prompt += f"<|im_start|>assistant\nThe clinical evidence points to option {ex['answer_idx']}.<|im_end|>\n"
            prompt += f"<|im_start|>user\nQuestion: {question}\nOptions: {options}\nLet's think step by step:<|im_end|>\n<|im_start|>assistant\n"
            return prompt

        elif strategy == PromptStrategy.TOT:
            return (f"{base_system}Use a Tree of Thoughts approach:\n"
                    "1. Explore 3 different clinical reasoning paths or differential diagnoses.\n"
                    "2. Evaluate the evidence for each and identify the most likely path.\n"
                    "3. Conclude with the final answer index (0, 1, 2, or 3) as 'Final Answer: [INDEX]'.<|im_end|>\n"
                    f"<|im_start|>user\nQuestion: {question}\nOptions: {options}\nThinking Process:<|im_end|>\n<|im_start|>assistant\n")
        
        return f"{base_system}<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"

# --- Shared Utilities ---

def get_processed_data_path(dataset_name: str, split: str) -> Path:
    return DATA_PATH / "processed" / f"{dataset_name}_{split}.jsonl"

def load_jsonl(path: Path, limit: Optional[int] = None) -> List[Dict]:
    if not path.exists():
        logger.warning(f"Path {path} does not exist.")
        return []
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data

# --- Modal Task Functions ---

@app.function(
    image=base_image, 
    volumes={DATA_PATH: volume}, 
    timeout=3600,
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def download_and_preprocess():
    """Ingests medical datasets from Hugging Face into a standardized schema."""
    from datasets import load_dataset
    
    os.makedirs(DATA_PATH / "raw", exist_ok=True)
    os.makedirs(DATA_PATH / "processed", exist_ok=True)

    datasets_to_load = [
        ("openlifescienceai/medmcqa", None, "medmcqa"),
        ("qiaojin/pubmed_qa", "pqa_labeled", "pubmed_qa"),
        ("openlifescienceai/MedQA-USMLE-4-options-hf", None, "med_qa"),
        ("cais/mmlu", "anatomy", "mmlu_anatomy"),
        ("cais/mmlu", "clinical_knowledge", "mmlu_clinical"),
    ]

    for hf_path, subset, internal_name in datasets_to_load:
        logger.info(f"Processing {internal_name}...")
        for split in ["train", "test", "validation"]:
            try:
                ds = load_dataset(hf_path, subset, split=split) if subset else load_dataset(hf_path, split=split)
                output_file = get_processed_data_path(internal_name, split)
                
                with open(output_file, "w", encoding="utf-8") as f:
                    for i, row in enumerate(ds):
                        q = row.get("question") or row.get("text", "")
                        opts = row.get("options") or row.get("choices")
                        if isinstance(opts, dict) and "A" in opts:
                            opts = [opts["A"], opts["B"], opts["C"], opts["D"]]
                        
                        ans = row.get("answer") or row.get("answer_idx") or row.get("label", -1)
                        if isinstance(ans, str) and ans in "ABCD":
                            ans = ord(ans) - ord("A")
                        elif isinstance(ans, str) and ans.isdigit():
                            ans = int(ans)
                        
                        f.write(json.dumps({
                            "id": f"{internal_name}_{split}_{i}",
                            "question": q,
                            "options": opts,
                            "answer_idx": ans,
                            "source": internal_name,
                            "metadata": {"subset": subset, "split": split}
                        }) + "\n")
                logger.info(f"Saved {len(ds)} records to {internal_name}_{split}")
            except Exception as e:
                logger.debug(f"Skipping {internal_name} {split}: {e}")

    volume.commit()

@app.function(
    image=base_image, 
    volumes={DATA_PATH: volume}, 
    timeout=7200,
    secrets=[
        modal.Secret.from_name("my-huggingface-secret"),
        modal.Secret.from_name("openrouter-secret")
    ]
)
def run_eval(strategy: str, model_id: str, num_samples: int = 100):
    """Executes evaluation using OpenRouter with async batching for high throughput."""
    import asyncio
    import httpx
    import numpy as np
    from tqdm import tqdm
    from sklearn.metrics import f1_score
    
    logger.info(f"Evaluator Initialized (OpenRouter): Strategy={strategy}, Model={model_id}")
    
    # OpenRouter API Configuration
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found. Please set the openrouter-secret in Modal.")

    # 1. Concurrency Control (10 at a time)
    semaphore = asyncio.Semaphore(10)
    
    async def call_openrouter(prompt: str, client: httpx.AsyncClient):
        async with semaphore:
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "top_p": 0.95,
                "max_tokens": 2048, # Increased to accommodate reasoning + answer (ToT/CoT)
            }
            headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://modal.com/medqa-research",
                "X-Title": "MedQA Research Suite"
            }
            
            for attempt in range(3):
                try:
                    resp = await client.post(
                        "https://openrouter.ai/api/v1/chat/completions",
                        json=payload,
                        headers=headers,
                        timeout=60.0
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if "error" in data:
                        logger.error(f"OpenRouter Error: {data['error']}")
                        return "ERROR: API Call Failed"

                    content = data["choices"][0]["message"].get("content")
                    reasoning = data["choices"][0]["message"].get("reasoning")
                    
                    final_output = content or reasoning
                    if final_output is None:
                        logger.warning(f"Empty content/reasoning received for {model_id}. Response: {data}")
                        return "ERROR: Empty Content"
                        
                    return final_output.strip()
                except Exception as e:
                    if attempt == 2:
                        logger.error(f"Failed after 3 attempts: {str(e)}")
                        return "ERROR: API Call Failed"
                    await asyncio.sleep(2 ** attempt)

    # 2. Setup CUBE Benchmark
    from cube_medqa import MedQABenchmark
    data_path = get_processed_data_path("med_qa", "test")
    if not data_path.exists():
        raise FileNotFoundError(f"MedQA test data not found at {data_path}. Run download-and-preprocess first.")
    
    benchmark = MedQABenchmark(str(data_path))
    num_to_run = min(num_samples, len(benchmark))
    
    few_shot_context = []
    if "few_shot" in strategy:
        few_shot_context = load_jsonl(get_processed_data_path("med_qa", "train"), limit=5)

    # 3. Execution Loop (CUBE-Compliant Async Batching)
    async def run_cube_batch():
        async with httpx.AsyncClient() as client:
            records = []
            for i in range(num_to_run):
                task = benchmark.get_task(i)
                obs = task.reset()
                full_prompt = PromptManager.construct_prompt(strategy, obs["question"], obs["options"], few_shot_context)
                records.append({
                    "task": task,
                    "prompt": full_prompt,
                    "future": call_openrouter(full_prompt, client)
                })
            
            # Dispatch all API calls concurrently
            results = await asyncio.gather(*[r["future"] for r in records])
            for r, output in zip(records, results):
                r["raw_output"] = output
            return records

    batch_results = asyncio.run(run_cube_batch())

    # 4. Parsing & CUBE Step Interaction
    results = []
    y_true, y_pred, confs = [], [], []

    for item in batch_results:
        task = item["task"]
        raw_output = item["raw_output"]
        confidence = 0.95  # Heuristic confidence

        # Parsing Logic
        cleaned = re.sub(r'<think>.*?</think>', '', raw_output, flags=re.DOTALL | re.IGNORECASE).strip()
        final_match = re.search(r'Final Answer:\s*([0-3])', cleaned, re.IGNORECASE)
        if final_match:
            pred_idx = int(final_match.group(1))
        else:
            digits = re.findall(r'[0-3]', cleaned)
            pred_idx = int(digits[-1]) if digits else -1
                
        # CUBE Step: Submitting the answer to the task object
        _, reward, done, info = task.step({"type": "answer", "idx": pred_idx})
        
        y_pred.append(pred_idx)
        y_true.append(task._data['answer_idx'])
        confs.append(confidence)
        
        results.append({
            "task_id": task._task_id,
            "raw_output": raw_output,
            "pred": pred_idx,
            "truth": task._data['answer_idx'],
            "reward": reward,
            "match": info.get("correct", False)
        })

    # 6. Metric Synthesis
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    acc = np.mean(y_true_np == y_pred_np)
    f1 = f1_score(y_true_np, y_pred_np, average='macro', labels=[0, 1, 2, 3])
    
    # Optimized ECE Calculation
    bin_boundaries = np.linspace(0, 1, 11)
    ece = 0
    confs_np = np.array(confs)
    for i in range(10):
        bin_idx = (confs_np > bin_boundaries[i]) & (confs_np <= bin_boundaries[i+1])
        if np.any(bin_idx):
            bin_acc = np.mean(y_pred_np[bin_idx] == y_true_np[bin_idx])
            bin_conf = np.mean(confs_np[bin_idx])
            ece += np.abs(bin_acc - bin_conf) * np.sum(bin_idx) / len(confs_np)
    
    logger.info(f"Summary [{strategy}]: Acc={acc:.4f}, F1={f1:.4f}, ECE={ece:.4f}")
    
    # 7. Persistence
    os.makedirs(RESULTS_PATH, exist_ok=True)
    report_tag = f"eval_{strategy}_{model_id.replace('/', '_')}.json"
    with open(RESULTS_PATH / report_tag, "w") as f:
        json.dump({
            "metadata": {"strategy": strategy, "model": model_id, "samples": num_samples, "engine": "openrouter-async"},
            "metrics": {"accuracy": float(acc), "f1_macro": float(f1), "ece": float(ece)},
            "details": results
        }, f, indent=2)
    
    volume.commit()
    return {"accuracy": acc, "f1": f1, "ece": ece}

@app.function(
    image=base_image, 
    volumes={DATA_PATH: volume}, 
    timeout=7200,
    secrets=[modal.Secret.from_name("my-huggingface-secret")]
)
def run_cube_benchmark(benchmark_name="medqa"):
    """Runs the CUBE-compliant benchmark for medical QA tools."""
    import asyncio
    from medical_mcp_server import app as mcp_app
    from cube_medqa import MedQABenchmark
    from mcp.client.session import ClientSession
    from mcp.client.stdio import stdio_client
    
    data_file = DATA_PATH / "processed" / "med_qa_test.jsonl"
    if not data_file.exists():
        return "Error: Processed MedQA test data not found."

    benchmark = MedQABenchmark(str(data_file))
    logger.info(f"Loaded CUBE benchmark: {len(benchmark)} tasks.")

    async def run_task(task_idx):
        task = benchmark.get_task(task_idx)
        observation = task.reset()
        
        server_params = {"command": "python", "args": ["medical_mcp_server.py"]}
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # Mock tool call for demo
                tool_result = await session.call_tool("pubmed_search", {"query": observation['question'][:20]})
                obs, reward, done, info = task.step({"type": "answer", "idx": 0})
                return reward, info

    async def run_demonstration():
        results = []
        for i in range(2): # Demonstration limit
            reward, info = await run_task(i)
            results.append({"task_index": i, "reward": reward, "info": info})
        return results

    loop = asyncio.get_event_loop()
    final_results = loop.run_until_complete(run_demonstration())
    logger.info(f"CUBE Demonstration Finished: {final_results}")
    return f"Completed {len(final_results)} CUBE tasks."

@app.local_entrypoint()
def zero_shot(model: str, samples: int = 100):
    """Run Zero-Shot evaluation."""
    run_eval.remote(strategy=PromptStrategy.ZERO_SHOT, model_id=model, num_samples=samples)

@app.local_entrypoint()
def few_shot(model: str, samples: int = 100):
    """Run Few-Shot (5 examples) evaluation."""
    run_eval.remote(strategy=PromptStrategy.FEW_SHOT, model_id=model, num_samples=samples)

@app.local_entrypoint()
def cot(model: str, samples: int = 100):
    """Run Chain-of-Thought evaluation."""
    run_eval.remote(strategy=PromptStrategy.COT, model_id=model, num_samples=samples)

@app.local_entrypoint()
def cot_few_shot(model: str, samples: int = 100):
    """Run CoT with Few-Shot examples evaluation."""
    run_eval.remote(strategy=PromptStrategy.COT_FEW_SHOT, model_id=model, num_samples=samples)

@app.local_entrypoint()
def tree_of_thoughts(model: str, samples: int = 100):
    """Run Tree-of-Thoughts evaluation."""
    run_eval.remote(strategy=PromptStrategy.TOT, model_id=model, num_samples=samples)
