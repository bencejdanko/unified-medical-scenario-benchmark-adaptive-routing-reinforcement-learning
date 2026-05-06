"""Run the unified medical scenario benchmark and return per-episode rewards.

The runner is intentionally router-free: one model is used for task completions,
and one judge model is used for HealthBench rubric grading. Results are written
incrementally as JSONL so failed test episodes can later seed an episodic memory
buffer without re-running the benchmark.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse

import requests
from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI

from healthbench_cube.healthbench_eval import HealthBenchEval, RubricItem, parse_json_to_dict as parse_healthbench_json
from healthbench_cube.types import MessageList, SamplerBase, SamplerResponse

load_dotenv()

DEFAULT_DATASET = "bdanko/unified_medical_scenario_benchmark"
DEFAULT_MODEL = "google/gemini-3-flash-preview"
DEFAULT_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_FHIR_BASE = "http://localhost:8080/fhir/"
DEFAULT_RESET_URL = "http://localhost:8080/reset"

ROUTER_MODEL_LABELS = [
    "google/gemini-3-flash-preview",
    "openai/gpt-oss-120b:nitro",
    "google/gemma-4-31b-it",
]
ROUTER_PROMPT_LABELS = ["mmlu_medical_json", "healthbench_default", "medagentbench_fhir"]
ROUTER_TOOL_LABELS = ["none", "fhir"]
FALLBACK_MODEL_BY_BENCHMARK = {
    "mmlu_medical": "openai/gpt-oss-120b:nitro",
    "healthbench": "google/gemini-3-flash-preview",
    "medagentbench": "google/gemma-4-31b-it",
}

MEDAGENTBENCH_PROMPT = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""


@dataclass
class EpisodeResult:
    run_id: str
    split: str
    benchmark: str
    task_id: str
    source_id: str
    model: str
    judge_model: str | None
    reward: float | None
    done: bool
    scorer: str
    prediction: Any = None
    target: Any = None
    error: str | None = None
    latency_s: float = 0.0
    prompt_messages: list[dict[str, Any]] = field(default_factory=list)
    transcript: list[dict[str, Any]] = field(default_factory=list)
    reward_details: dict[str, Any] = field(default_factory=dict)
    router_context: dict[str, Any] = field(default_factory=dict)
    source_metadata: dict[str, Any] = field(default_factory=dict)
    episodic_memory_candidate: bool = False


class OpenRouterSampler(SamplerBase):
    def __init__(
        self,
        client: OpenAI,
        model: str,
        temperature: float,
        max_tokens: int,
        api_semaphore: threading.BoundedSemaphore | None = None,
    ):
        self.client = client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.api_semaphore = api_semaphore
        self.usage_records: list[dict[str, Any]] = []
        self._usage_lock = threading.Lock()

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        limiter = getattr(self, "api_semaphore", None)
        context = limiter if limiter is not None else contextlib.nullcontext()
        with context:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=message_list,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        content = response.choices[0].message.content or ""
        usage = response.usage.model_dump() if response.usage else {}
        with self._usage_lock:
            self.usage_records.append(usage)
        return SamplerResponse(
            response_text=content,
            response_metadata={"usage": usage},
            actual_queried_message_list=message_list,
        )


def now_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def parse_json_field(row: dict[str, Any], key: str, default: Any) -> Any:
    value = row.get(key)
    if value in (None, ""):
        return default
    return json.loads(value)


def normalize_option(text: str) -> str:
    text = (text or "").strip().upper()
    if not text:
        return ""
    match = re.search(r"\b([A-D])\b", text)
    return match.group(1) if match else text[0]


def extract_json(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return {}
    return {}


def call_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    api_semaphore: threading.BoundedSemaphore | None = None,
    retries: int = 4,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            context = api_semaphore if api_semaphore is not None else contextlib.nullcontext()
            with context:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
            content = response.choices[0].message.content or ""
            usage = response.usage.model_dump() if response.usage else {}
            return content, {"usage": usage}
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError(last_error)


def write_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(item, ensure_ascii=False, default=str) + "\n")


def source_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dataset": row["source_dataset"],
        "config": row["source_config"],
        "source_split": row["source_split"],
        "source_id": row["source_id"],
        "source_index": row["source_index"],
        "scenario_type": row["scenario_type"],
        "task_family": row["task_family"],
        "is_hard": row["is_hard"],
        "original_record_sha256": row["original_record_sha256"],
    }


def router_context(row: dict[str, Any]) -> dict[str, Any]:
    metadata = parse_json_field(row, "router_metadata_json", {})
    return {
        "benchmark": row["benchmark"],
        "scenario_type": row["scenario_type"],
        "task_family": row["task_family"],
        "is_hard": row["is_hard"],
        **metadata,
    }


def router_text(row: dict[str, Any]) -> str:
    parts = [
        f"benchmark: {row['benchmark']}",
        f"scenario_type: {row.get('scenario_type', '')}",
        f"task_family: {row.get('task_family', '')}",
        f"is_hard: {row.get('is_hard', '')}",
    ]
    if row["benchmark"] == "mmlu_medical":
        choices = parse_json_field(row, "choices_json", [])
        parts.append(f"question: {row.get('question', '')}")
        parts.append("choices: " + " ".join(f"{'ABCD'[idx]}. {choice}" for idx, choice in enumerate(choices)))
    elif row["benchmark"] == "healthbench":
        prompt = parse_json_field(row, "messages_json", [])
        parts.append("request: " + " ".join(str(item.get("content", "")) for item in prompt if item.get("role") == "user"))
    elif row["benchmark"] == "medagentbench":
        parts.append(f"context: {row.get('context', '')}")
        parts.append(f"instruction: {row.get('instruction', '')}")
    return "\n".join(parts)


class DistilBertRouterRuntime:
    def __init__(self, model_path: str):
        import torch
        from torch import nn
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        config = AutoConfig.from_pretrained(model_path)

        class RouterModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = AutoModel.from_config(config)
                self.dropout = nn.Dropout(config.seq_classif_dropout)
                self.model_head = nn.Linear(config.dim, 3)
                self.tool_head = nn.Linear(config.dim, 2)
                self.prompt_head = nn.Linear(config.dim, 3)

            def forward(self, **inputs):
                outputs = self.encoder(**inputs)
                pooled = self.dropout(outputs.last_hidden_state[:, 0])
                return {
                    "model": self.model_head(pooled),
                    "tool": self.tool_head(pooled),
                    "prompt": self.prompt_head(pooled),
                }

        self.model = RouterModule().to(self.device)
        state = torch.load(Path(model_path) / "pytorch_model.bin", map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        labels_path = Path(model_path) / "router_labels.json"
        labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
        self.model_labels = labels.get("model_labels", ROUTER_MODEL_LABELS)
        self.prompt_labels = labels.get("prompt_labels", ROUTER_PROMPT_LABELS)
        self.tool_labels = labels.get("tool_labels", ROUTER_TOOL_LABELS)

    def route(self, row: dict[str, Any]) -> dict[str, Any]:
        encoded = self.tokenizer(router_text(row), truncation=True, padding=True, max_length=384, return_tensors="pt")
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with self.torch.no_grad():
            logits = self.model(**encoded)
        model_probs = self.torch.softmax(logits["model"], dim=-1)[0]
        tool_probs = self.torch.softmax(logits["tool"], dim=-1)[0]
        prompt_probs = self.torch.softmax(logits["prompt"], dim=-1)[0]
        model_idx = int(logits["model"].argmax(dim=-1).item())
        tool_idx = int(logits["tool"].argmax(dim=-1).item())
        prompt_idx = int(logits["prompt"].argmax(dim=-1).item())
        return {
            "selected_model": self.model_labels[model_idx],
            "selected_tool": self.tool_labels[tool_idx],
            "selected_prompt": self.prompt_labels[prompt_idx],
            "router_confidence": {
                "model": float(model_probs[model_idx].item()),
                "tool": float(tool_probs[tool_idx].item()),
                "prompt": float(prompt_probs[prompt_idx].item()),
            },
            "router_probabilities": {
                "model": {label: float(model_probs[idx].item()) for idx, label in enumerate(self.model_labels)},
                "tool": {label: float(tool_probs[idx].item()) for idx, label in enumerate(self.tool_labels)},
                "prompt": {label: float(prompt_probs[idx].item()) for idx, label in enumerate(self.prompt_labels)},
            },
            "router_source": "distilbert",
        }


def fallback_route(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "selected_model": FALLBACK_MODEL_BY_BENCHMARK[row["benchmark"]],
        "selected_tool": "fhir" if row["benchmark"] == "medagentbench" else "none",
        "selected_prompt": {
            "mmlu_medical": "mmlu_medical_json",
            "healthbench": "healthbench_default",
            "medagentbench": "medagentbench_fhir",
        }[row["benchmark"]],
        "router_confidence": {"model": 1.0, "tool": 1.0, "prompt": 1.0},
        "router_source": "fallback_policy",
    }


def route_row(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    router = getattr(args, "router_runtime", None)
    if router is not None:
        try:
            return router.route(row)
        except Exception as exc:
            route = fallback_route(row)
            route["router_error"] = str(exc)
            return route
    return fallback_route(row)


class EpisodicMemoryRuntime:
    def __init__(
        self,
        model_path: str,
        top_k: int,
        dataset_path: str | None = None,
        chunks_path: str | None = None,
        embeddings_path: str | None = None,
    ):
        import numpy as np
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.model = SentenceTransformer(model_path)
        self.top_k = top_k
        if chunks_path and embeddings_path:
            self.rows = []
            with Path(chunks_path).open() as f:
                for line in f:
                    self.rows.append(json.loads(line))
            self.embeddings = np.load(embeddings_path).astype(np.float32)
            return

        if not dataset_path:
            raise ValueError("Either chunks/embeddings paths or dataset_path must be provided for episodic memory.")
        from datasets import load_from_disk

        ds = load_from_disk(dataset_path)
        split = ds["train"] if hasattr(ds, "keys") and "train" in ds else ds
        self.rows = [dict(row) for row in split if row.get("text")]
        texts = [row["text"] for row in self.rows]
        self.embeddings = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False) if texts else None

    def retrieve(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        if not self.rows or self.embeddings is None:
            return []
        query = router_text(row)
        query_vec = self.model.encode([query], normalize_embeddings=True, show_progress_bar=False)[0]
        scores = self.embeddings @ query_vec
        order = self.np.argsort(-scores)[: self.top_k]
        memories = []
        for idx in order:
            memory = dict(self.rows[int(idx)])
            memory["score"] = float(scores[int(idx)])
            memories.append(memory)
        return memories


def episodic_memory_block(row: dict[str, Any], args: argparse.Namespace) -> str:
    memory = getattr(args, "episodic_memory_runtime", None)
    if memory is None:
        return ""
    try:
        memories = memory.retrieve(row)
    except Exception:
        return ""
    row["_episodic_memory_retrieval"] = [
        {
            "id": item.get("id"),
            "task_id": item.get("task_id"),
            "benchmark": item.get("benchmark"),
            "model": item.get("model"),
            "reward": item.get("reward"),
            "score": item.get("score"),
            "chunk_index": item.get("chunk_index"),
        }
        for item in memories
    ]
    if not memories:
        return ""
    lines = [
        "Relevant lessons from prior failed attempts. Use these as general guidance only; do not copy task-specific answers."
    ]
    for idx, item in enumerate(memories, start=1):
        text = str(item.get("text", "")).strip()
        if len(text) > 1200:
            text = text[:1200] + "..."
        lines.append(f"Memory {idx} (score={item.get('score', 0.0):.3f}):\n{text}")
    return "\n\n".join(lines)


def mmlu_messages(row: dict[str, Any], memory_block: str = "") -> list[dict[str, str]]:
    choices = parse_json_field(row, "choices_json", [])
    body = row["question"] + "\n\n"
    for idx, choice in enumerate(choices):
        body += f"{'ABCD'[idx]}. {choice}\n"
    system_content = (
        "You are a medical expert. Return ONLY valid JSON in this exact format: "
        "{\"answer\":\"A\",\"confidence\":0.0}. The answer must be A, B, C, or D."
    )
    if memory_block:
        system_content += "\n\n" + memory_block
    return [
        {
            "role": "system",
            "content": system_content,
        },
        {"role": "user", "content": body},
    ]


def run_mmlu(row: dict[str, Any], client: OpenAI, args: argparse.Namespace, run_id: str) -> EpisodeResult:
    messages = mmlu_messages(row, episodic_memory_block(row, args))
    start = time.time()
    raw, meta = call_chat(
        client,
        args.model,
        messages,
        temperature=0.0,
        max_tokens=128,
        api_semaphore=getattr(args, "api_semaphore", None),
    )
    parsed = extract_json(raw)
    predicted = normalize_option(str(parsed.get("answer", raw)))
    target = row["correct_option"]
    reward = 1.0 if predicted == target else 0.0
    return EpisodeResult(
        run_id=run_id,
        split=args.split,
        benchmark=row["benchmark"],
        task_id=row["id"],
        source_id=row["source_id"],
        model=args.model,
        judge_model=None,
        reward=reward,
        done=True,
        scorer="multiple_choice_exact_match",
        prediction={"answer": predicted, "raw": raw, "metadata": meta},
        target={"correct_option": target, "correct_answer": row["correct_answer"]},
        latency_s=time.time() - start,
        prompt_messages=messages,
        transcript=messages + [{"role": "assistant", "content": raw}],
        reward_details={"confidence": parsed.get("confidence")},
        router_context=router_context(row),
        source_metadata=source_metadata(row),
        episodic_memory_candidate=reward == 0.0,
    )


def run_healthbench(row: dict[str, Any], client: OpenAI, judge_client: OpenAI, args: argparse.Namespace, run_id: str) -> EpisodeResult:
    prompt = parse_json_field(row, "messages_json", [])
    memory_block = episodic_memory_block(row, args)
    if memory_block:
        prompt = [{"role": "system", "content": memory_block}] + prompt
    rubrics = [RubricItem.from_dict(item) for item in parse_json_field(row, "rubrics_json", [])]
    example_tags = parse_json_field(row, "router_metadata_json", {}).get("example_tags", [])
    start = time.time()
    response_start = time.time()
    response_text, response_meta = call_chat(
        client,
        args.model,
        prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        api_semaphore=getattr(args, "api_semaphore", None),
    )
    response_latency_s = time.time() - response_start
    grader_sampler = OpenRouterSampler(
        judge_client,
        args.judge_model,
        temperature=0.0,
        max_tokens=args.judge_max_tokens,
        api_semaphore=getattr(args, "api_semaphore", None),
    )
    grader = HealthBenchEval(
        grader_model=grader_sampler,
        num_examples=1,
        n_threads=args.healthbench_threads,
        subset_name="hard",
    )
    original_grade_sample = grader.grade_sample

    def bounded_grade_sample(
        prompt: list[dict[str, str]],
        response_text: str,
        example_tags: list[str],
        rubric_items: list[RubricItem],
    ) -> tuple[dict, str, list[dict]]:
        original_grader_model = grader.grader_model

        class BoundedGrader(SamplerBase):
            def __call__(self, message_list: MessageList) -> SamplerResponse:
                last_response: SamplerResponse | None = None
                for _ in range(args.healthbench_grader_retries):
                    last_response = original_grader_model(message_list)
                    parsed = parse_healthbench_json(last_response.response_text)
                    if isinstance(parsed.get("criteria_met"), bool):
                        return last_response
                    print("Grading failed due to bad JSON output, retrying...")
                fallback = {
                    "explanation": (
                        "Automatic fallback after repeated invalid grader JSON. "
                        "The rubric item is marked unmet to avoid an infinite retry loop."
                    ),
                    "criteria_met": False,
                }
                return SamplerResponse(
                    response_text=json.dumps(fallback),
                    response_metadata={"fallback_after_invalid_json": True},
                    actual_queried_message_list=message_list,
                )

        grader.grader_model = BoundedGrader()
        try:
            return original_grade_sample(prompt, response_text, example_tags, rubric_items)
        finally:
            grader.grader_model = original_grader_model

    grader.grade_sample = bounded_grade_sample
    judge_start = time.time()
    metrics, explanation, rubric_grades = grader.grade_sample(
        prompt=prompt,
        response_text=response_text,
        example_tags=example_tags,
        rubric_items=rubrics,
    )
    judge_latency_s = time.time() - judge_start
    reward = float(metrics["overall_score"])
    total_latency_s = time.time() - start
    return EpisodeResult(
        run_id=run_id,
        split=args.split,
        benchmark=row["benchmark"],
        task_id=row["id"],
        source_id=row["source_id"],
        model=args.model,
        judge_model=args.judge_model,
        reward=reward,
        done=True,
        scorer="healthbench_weighted_rubric",
        prediction={"response_text": response_text, "metadata": response_meta},
        target={"rubrics": [item.to_dict() for item in rubrics]},
        latency_s=total_latency_s,
        prompt_messages=prompt,
        transcript=prompt + [{"role": "assistant", "content": response_text}],
        reward_details={
            "metrics": metrics,
            "rubric_grades": rubric_grades,
            "readable_explanation": explanation,
            "judge_usage": grader_sampler.usage_records,
            "latency_s": {
                "response": response_latency_s,
                "judge": judge_latency_s,
                "total": total_latency_s,
            },
        },
        router_context=router_context(row),
        source_metadata=source_metadata(row),
        episodic_memory_candidate=reward == 0.0,
    )


def medagent_headers() -> dict[str, str]:
    api_key = os.getenv("MEDAGENTBENCH_API_KEY")
    return {"X-API-KEY": api_key} if api_key else {}


def append_format_json(url: str) -> str:
    if "_format=" in url:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}_format=json"


def request_fhir(method: str, url: str, payload: Any | None = None, timeout: float = 60.0) -> dict[str, Any]:
    headers = medagent_headers()
    if method == "GET":
        response = requests.get(append_format_json(url), headers=headers, timeout=timeout)
    elif method == "POST":
        response = requests.post(url, json=payload, headers=headers, timeout=timeout)
    else:
        raise ValueError(f"Unsupported method: {method}")
    content_type = response.headers.get("Content-Type", "")
    body: Any
    if "json" in content_type:
        body = response.json()
    else:
        body = response.text
    return {"status_code": response.status_code, "data": body, "ok": response.ok}


def reset_medagentbench(reset_url: str, timeout: float) -> dict[str, Any]:
    response = requests.post(reset_url, headers=medagent_headers(), timeout=timeout)
    try:
        body = response.json()
    except Exception:
        body = response.text
    return {"status_code": response.status_code, "data": body, "ok": response.ok}


@contextlib.contextmanager
def patch_requests_auth_for_refsol(fhir_base: str):
    original_request = requests.sessions.Session.request
    parsed_base = urlparse(fhir_base)

    def patched(self, method, url, **kwargs):
        parsed_url = urlparse(str(url))
        if parsed_url.netloc == parsed_base.netloc:
            headers = dict(kwargs.pop("headers", None) or {})
            headers.update(medagent_headers())
            kwargs["headers"] = headers
        return original_request(self, method, url, **kwargs)

    requests.sessions.Session.request = patched
    try:
        yield
    finally:
        requests.sessions.Session.request = original_request


def load_medagentbench_eval():
    root = Path(__file__).parent / "medagentbench_cube"
    root_str = str(root.resolve())
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    try:
        module = importlib.import_module("src.server.tasks.medagentbench.eval")
        output_module = importlib.import_module("src.typings.output")
    except ModuleNotFoundError as exc:
        if "refsol" in str(exc):
            return None, None, "Missing official refsol.py at medagentbench_cube/src/server/tasks/medagentbench/refsol.py"
        return None, None, str(exc)
    except Exception as exc:
        return None, None, str(exc)
    return module.eval, output_module.TaskOutput, None


def parse_finish(content: str) -> str | None:
    text = content.strip()
    if text.startswith("FINISH(") and text.endswith(")"):
        return text[len("FINISH(") : -1]
    return None


def medagentbench_judge_messages(row: dict[str, Any], finish_result: str | None, official_pass: bool | None) -> list[dict[str, str]]:
    expected = parse_json_field(row, "acceptable_answers_json", [])
    body = {
        "task_id": row["source_id"],
        "instruction": row["instruction"],
        "context": row["context"],
        "expected_outputs": expected,
        "official_refsol_pass": official_pass,
        "model_finish_payload": finish_result,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are judging a MedAgentBench task. Compare the model's FINISH payload "
                "against the expected outputs and official refsol pass flag. Return ONLY valid JSON "
                "with keys score, success, explanation. score must be 1.0 for successful completion "
                "and 0.0 for unsuccessful completion. Treat semantically correct numeric/list answers "
                "as success even if extra prose is present, but do not ignore wrong values, wrong orders, "
                "or missing required actions."
            ),
        },
        {"role": "user", "content": json.dumps(body, ensure_ascii=False, default=str)},
    ]


def run_medagentbench_judge(
    row: dict[str, Any],
    finish_result: str | None,
    official_pass: bool | None,
    client: OpenAI,
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    messages = medagentbench_judge_messages(row, finish_result, official_pass)
    raw, meta = call_chat(
        client,
        args.medagentbench_judge_model,
        messages,
        temperature=0.0,
        max_tokens=args.medagentbench_judge_max_tokens,
        api_semaphore=getattr(args, "api_semaphore", None),
    )
    parsed = extract_json(raw)
    score = parsed.get("score")
    if isinstance(score, bool):
        score = 1.0 if score else 0.0
    if isinstance(score, int | float):
        score = 1.0 if float(score) >= 0.5 else 0.0
    else:
        success = parsed.get("success")
        score = 1.0 if success is True else 0.0
    return {
        "score": score,
        "success": bool(score),
        "explanation": parsed.get("explanation"),
        "raw": raw,
    }, meta


def run_medagentbench(row: dict[str, Any], client: OpenAI, args: argparse.Namespace, run_id: str) -> EpisodeResult:
    fhir_base = args.medagentbench_fhir_base.rstrip("/")
    reset_info = None
    reset_latency_s = 0.0
    if args.reset_medagentbench:
        reset_start = time.time()
        reset_info = reset_medagentbench(args.medagentbench_reset_url, args.medagentbench_reset_timeout)
        reset_latency_s = time.time() - reset_start
    tools = parse_json_field(row, "tools_json", [])
    memory_block = episodic_memory_block(row, args)
    context = row["context"] + ("\n\n" + memory_block if memory_block else "")
    messages = [
        {
            "role": "user",
            "content": MEDAGENTBENCH_PROMPT.format(
                api_base=fhir_base,
                functions=json.dumps(tools),
                context=context,
                question=row["instruction"],
            ),
        }
    ]
    transcript = list(messages)
    finish_result = None
    status = "task limit reached"
    llm_usage: list[dict[str, Any]] = []
    start = time.time()
    agent_start = time.time()
    for _ in range(args.medagentbench_max_rounds):
        raw, meta = call_chat(
            client,
            args.model,
            messages,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_semaphore=getattr(args, "api_semaphore", None),
        )
        usage = meta.get("usage")
        if isinstance(usage, dict):
            llm_usage.append(usage)
        action_text = raw.strip().replace("```tool_code", "").replace("```", "").strip()
        messages.append({"role": "assistant", "content": action_text})
        transcript.append({"role": "assistant", "content": action_text})
        finish_result = parse_finish(action_text)
        if finish_result is not None:
            status = "completed"
            break
        if action_text.startswith("GET"):
            url = action_text[3:].strip()
            obs = request_fhir("GET", url, timeout=args.medagentbench_http_timeout)
            obs_text = (
                f"Here is the response from the GET request:\n{obs['data']}. "
                "Please call FINISH if you have got answers for all the questions and finished all the requested tasks"
            )
        elif action_text.startswith("POST"):
            parts = action_text.split("\n", 1)
            url = parts[0][4:].strip()
            payload_text = parts[1] if len(parts) > 1 else "{}"
            try:
                payload = json.loads(payload_text)
                obs = request_fhir("POST", url, payload=payload, timeout=args.medagentbench_http_timeout)
                obs_text = (
                    "POST request executed. "
                    f"Status: {obs['status_code']}. Response: {obs['data']}. "
                    "Please call FINISH if you have got answers for all the questions and finished all the requested tasks"
                )
            except Exception as exc:
                obs_text = f"Invalid POST request: {exc}"
        else:
            status = "agent invalid action"
            break
        user_obs = {"role": "user", "content": obs_text}
        messages.append(user_obs)
        transcript.append(user_obs)
    agent_latency_s = time.time() - agent_start

    eval_func, TaskOutputCls, eval_error = load_medagentbench_eval()
    reward = None
    official_pass = None
    scorer_latency_s = 0.0
    judge_latency_s = 0.0
    reward_details: dict[str, Any] = {"status": status, "reset": reset_info, "llm_usage": llm_usage}
    error = eval_error
    if eval_func is not None and TaskOutputCls is not None and finish_result is not None:
        task_output = TaskOutputCls(index=0, status=status, result=finish_result, history=[])
        case_data = parse_json_field(row, "original_record_json", {})
        scorer_start = time.time()
        with patch_requests_auth_for_refsol(fhir_base):
            passed = eval_func(case_data, task_output, fhir_base + "/")
        scorer_latency_s = time.time() - scorer_start
        official_pass = passed is True
        reward = 1.0 if passed is True else 0.0
        reward_details["official_pass"] = official_pass
        error = None
    elif finish_result is None and error is None:
        error = f"No FINISH action produced; status={status}"

    judge_model = None
    if args.medagentbench_judge and finish_result is not None:
        judge_start = time.time()
        judge_result, judge_meta = run_medagentbench_judge(row, finish_result, official_pass, client, args)
        judge_latency_s = time.time() - judge_start
        reward_details["judge"] = judge_result
        judge_usage = judge_meta.get("usage")
        if isinstance(judge_usage, dict):
            reward_details["judge_usage"] = [judge_usage]
        judge_model = args.medagentbench_judge_model
    total_latency_s = time.time() - start
    reward_details["latency_s"] = {
        "reset": reset_latency_s,
        "agent": agent_latency_s,
        "official_scorer": scorer_latency_s,
        "judge": judge_latency_s,
        "total": total_latency_s,
    }

    return EpisodeResult(
        run_id=run_id,
        split=args.split,
        benchmark=row["benchmark"],
        task_id=row["id"],
        source_id=row["source_id"],
        model=args.model,
        judge_model=judge_model,
        reward=reward,
        done=finish_result is not None,
        scorer="official_medagentbench_refsol" if eval_func else "unscored_missing_refsol",
        prediction={"finish_result": finish_result, "status": status},
        target={"acceptable_answers": parse_json_field(row, "acceptable_answers_json", [])},
        error=error,
        latency_s=total_latency_s,
        prompt_messages=messages[:1],
        transcript=transcript,
        reward_details=reward_details,
        router_context=router_context(row),
        source_metadata=source_metadata(row),
        episodic_memory_candidate=reward == 0.0,
    )


def existing_task_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids = set()
    with path.open() as f:
        for line in f:
            try:
                ids.add(json.loads(line)["task_id"])
            except Exception:
                pass
    return ids


def load_episode_results(path: Path) -> list[EpisodeResult]:
    if not path.exists():
        return []
    fields = set(EpisodeResult.__dataclass_fields__)
    results = []
    with path.open() as f:
        for line in f:
            try:
                item = json.loads(line)
                results.append(EpisodeResult(**{key: item[key] for key in fields if key in item}))
            except Exception:
                pass
    return results


def select_rows(rows: Iterable[dict[str, Any]], benchmarks: list[str], limit: int | None, limit_per_benchmark: int | None) -> list[dict[str, Any]]:
    allowed = set(benchmarks)
    selected = [row for row in rows if row["benchmark"] in allowed]
    if limit_per_benchmark is not None:
        counts: dict[str, int] = {}
        out = []
        for row in selected:
            count = counts.get(row["benchmark"], 0)
            if count < limit_per_benchmark:
                out.append(row)
                counts[row["benchmark"]] = count + 1
        selected = out
    if limit is not None:
        selected = selected[:limit]
    return selected


def summarize(results: list[EpisodeResult]) -> dict[str, Any]:
    by_benchmark: dict[str, list[EpisodeResult]] = {}
    for result in results:
        by_benchmark.setdefault(result.benchmark, []).append(result)
    summary = {}
    for benchmark, items in by_benchmark.items():
        scored = [x for x in items if x.reward is not None]
        judge_scored = [
            x.reward_details.get("judge", {}).get("score")
            for x in items
            if isinstance(x.reward_details, dict) and isinstance(x.reward_details.get("judge"), dict)
        ]
        judge_scored = [float(x) for x in judge_scored if isinstance(x, int | float)]
        latencies = [float(x.latency_s) for x in items if x.latency_s is not None]
        component_latencies: dict[str, list[float]] = {}
        for item in items:
            if not isinstance(item.reward_details, dict):
                continue
            latency_details = item.reward_details.get("latency_s")
            if not isinstance(latency_details, dict):
                continue
            for key, value in latency_details.items():
                if isinstance(value, int | float):
                    component_latencies.setdefault(key, []).append(float(value))
        summary[benchmark] = {
            "n": len(items),
            "scored_n": len(scored),
            "mean_reward": sum(float(x.reward) for x in scored) / len(scored) if scored else None,
            "failures": sum(1 for x in scored if x.reward == 0.0),
            "unscored": len(items) - len(scored),
            "mean_judge_reward": sum(judge_scored) / len(judge_scored) if judge_scored else None,
            "latency_s_total": sum(latencies) if latencies else 0.0,
            "latency_s_mean": sum(latencies) / len(latencies) if latencies else None,
            "latency_s_min": min(latencies) if latencies else None,
            "latency_s_max": max(latencies) if latencies else None,
            "latency_s_components_mean": {
                key: sum(values) / len(values)
                for key, values in component_latencies.items()
                if values
            },
            "latency_s_components_total": {
                key: sum(values)
                for key, values in component_latencies.items()
                if values
            },
        }
    return summary


def usage_from_result(result: EpisodeResult) -> dict[str, Any]:
    usages: list[dict[str, Any]] = []
    prediction = result.prediction if isinstance(result.prediction, dict) else {}
    metadata = prediction.get("metadata")
    if isinstance(metadata, dict) and isinstance(metadata.get("usage"), dict):
        usages.append(metadata["usage"])
    if isinstance(metadata, dict) and isinstance(metadata.get("usage"), list):
        usages.extend(item for item in metadata["usage"] if isinstance(item, dict))
    details = result.reward_details if isinstance(result.reward_details, dict) else {}
    llm_usage = details.get("llm_usage")
    if isinstance(llm_usage, list):
        usages.extend(item for item in llm_usage if isinstance(item, dict))
    metrics = details.get("metrics")
    if isinstance(metrics, dict):
        for value in metrics.values():
            if isinstance(value, dict) and isinstance(value.get("usage"), dict):
                usages.append(value["usage"])
    judge_usage = details.get("judge_usage")
    if isinstance(judge_usage, list):
        usages.extend(item for item in judge_usage if isinstance(item, dict))

    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "usage_records": 0,
    }
    for usage in usages:
        totals["usage_records"] += 1
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(key)
            if isinstance(value, int | float):
                totals[key] += int(value)
        cost = usage.get("cost")
        if isinstance(cost, int | float):
            totals["cost"] += float(cost)
    return totals


def summarize_usage(results: list[EpisodeResult]) -> dict[str, Any]:
    by_benchmark: dict[str, dict[str, Any]] = {}
    total = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost": 0.0,
        "usage_records": 0,
    }
    for result in results:
        usage = usage_from_result(result)
        bucket = by_benchmark.setdefault(
            result.benchmark,
            {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "cost": 0.0,
                "usage_records": 0,
            },
        )
        for key, value in usage.items():
            bucket[key] += value
            total[key] += value
    return {"total": total, "by_benchmark": by_benchmark}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_routing(results: list[EpisodeResult]) -> dict[str, Any]:
    routed = [result for result in results if isinstance(result.router_context, dict) and result.router_context.get("selected_model")]
    if not routed:
        return {}
    by_selected_model: dict[str, list[EpisodeResult]] = {}
    by_prompt: dict[str, int] = {}
    by_tool: dict[str, int] = {}
    by_source: dict[str, int] = {}
    confusion: dict[str, dict[str, int]] = {}
    memory_counts: list[float] = []
    memory_scores: list[float] = []
    confidences: dict[str, list[float]] = {"model": [], "tool": [], "prompt": []}

    for result in routed:
        context = result.router_context
        selected_model = str(context.get("selected_model"))
        by_selected_model.setdefault(selected_model, []).append(result)
        selected_prompt = str(context.get("selected_prompt", "unknown"))
        selected_tool = str(context.get("selected_tool", "unknown"))
        router_source = str(context.get("router_source", "unknown"))
        by_prompt[selected_prompt] = by_prompt.get(selected_prompt, 0) + 1
        by_tool[selected_tool] = by_tool.get(selected_tool, 0) + 1
        by_source[router_source] = by_source.get(router_source, 0) + 1
        confusion.setdefault(result.benchmark, {})
        confusion[result.benchmark][selected_model] = confusion[result.benchmark].get(selected_model, 0) + 1

        confidence = context.get("router_confidence")
        if isinstance(confidence, dict):
            for key in confidences:
                value = confidence.get(key)
                if isinstance(value, int | float):
                    confidences[key].append(float(value))

        memories = context.get("episodic_memory_retrieval")
        if isinstance(memories, list):
            memory_counts.append(float(len(memories)))
            for memory in memories:
                if isinstance(memory, dict) and isinstance(memory.get("score"), int | float):
                    memory_scores.append(float(memory["score"]))

    selected_model_summary = {}
    for selected_model, items in by_selected_model.items():
        scored = [item for item in items if item.reward is not None]
        latencies = [float(item.latency_s) for item in items if item.latency_s is not None]
        usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "usage_records": 0,
        }
        for item in items:
            item_usage = usage_from_result(item)
            for key, value in item_usage.items():
                usage[key] += value
        selected_model_summary[selected_model] = {
            "n": len(items),
            "scored_n": len(scored),
            "mean_reward": mean([float(item.reward) for item in scored]),
            "failures": sum(1 for item in scored if item.reward == 0.0),
            "latency_s_mean": mean(latencies),
            "usage": usage,
        }

    return {
        "routed_n": len(routed),
        "selected_model_counts": {key: len(value) for key, value in by_selected_model.items()},
        "selected_prompt_counts": by_prompt,
        "selected_tool_counts": by_tool,
        "router_source_counts": by_source,
        "benchmark_to_selected_model_counts": confusion,
        "selected_model_summary": selected_model_summary,
        "model_diversity": {
            "unique_selected_models": len(by_selected_model),
            "selected_model_share": {key: len(value) / len(routed) for key, value in by_selected_model.items()},
        },
        "router_confidence_mean": {key: mean(values) for key, values in confidences.items()},
        "episodic_memory": {
            "enabled_n": len(memory_counts),
            "mean_retrieved": mean(memory_counts),
            "mean_similarity": mean(memory_scores),
        },
    }


def run_one(row: dict[str, Any], client: OpenAI, judge_client: OpenAI, args: argparse.Namespace, run_id: str) -> EpisodeResult:
    try:
        original_model = args.model
        route: dict[str, Any] | None = None
        if args.routed:
            route = route_row(row, args)
            args = copy.copy(args)
            args.model = route["selected_model"]
        if row["benchmark"] == "mmlu_medical":
            result = run_mmlu(row, client, args, run_id)
        elif row["benchmark"] == "healthbench":
            result = run_healthbench(row, client, judge_client, args, run_id)
        elif row["benchmark"] == "medagentbench":
            result = run_medagentbench(row, client, args, run_id)
        else:
            raise ValueError(f"Unsupported benchmark: {row['benchmark']}")
        if route is not None:
            result.router_context = {
                **result.router_context,
                **route,
                "default_model": original_model,
                "episodic_memory_retrieval": row.get("_episodic_memory_retrieval", []),
            }
        return result
    except Exception as exc:
        return EpisodeResult(
            run_id=run_id,
            split=args.split,
            benchmark=row["benchmark"],
            task_id=row["id"],
            source_id=row["source_id"],
            model=args.model,
            judge_model=args.judge_model if row["benchmark"] == "healthbench" else None,
            reward=None,
            done=False,
            scorer="error",
            error=str(exc),
            router_context=router_context(row),
            source_metadata=source_metadata(row),
        )


def record_result(result: EpisodeResult, result_path: Path, memory_path: Path) -> None:
    result.episodic_memory_candidate = result.episodic_memory_candidate or result.reward == 0.0
    item = asdict(result)
    item["episode_hash"] = hashlib.sha256(
        json.dumps({"task_id": result.task_id, "model": result.model, "prediction": result.prediction}, default=str).encode("utf-8")
    ).hexdigest()
    write_jsonl(result_path, item)
    if result.episodic_memory_candidate:
        write_jsonl(memory_path, item)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Unified medical scenario evaluation runner")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="test")
    parser.add_argument("--benchmarks", nargs="+", default=["mmlu_medical", "healthbench", "medagentbench"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--limit-per-benchmark", type=int, default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--api-base", default=os.getenv("OPENROUTER_API_URL", DEFAULT_API_BASE))
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--judge-max-tokens", type=int, default=2048)
    parser.add_argument("--healthbench-threads", type=int, default=4)
    parser.add_argument("--healthbench-grader-retries", type=int, default=8)
    parser.add_argument("--parallelism", type=int, default=1, help="Max concurrent stateless benchmark episodes.")
    parser.add_argument(
        "--max-openrouter-requests",
        type=int,
        default=10,
        help="Global cap on concurrent OpenRouter requests, including HealthBench rubric grading.",
    )
    parser.add_argument("--medagentbench-fhir-base", default=os.getenv("MEDAGENTBENCH_FHIR_API_BASE", DEFAULT_FHIR_BASE))
    parser.add_argument("--medagentbench-reset-url", default=os.getenv("MEDAGENTBENCH_RESET_URL", DEFAULT_RESET_URL))
    parser.add_argument("--medagentbench-max-rounds", type=int, default=5)
    parser.add_argument("--medagentbench-http-timeout", type=float, default=60.0)
    parser.add_argument("--medagentbench-reset-timeout", type=float, default=240.0)
    parser.add_argument("--medagentbench-judge", action="store_true")
    parser.add_argument("--medagentbench-judge-model", default=DEFAULT_MODEL)
    parser.add_argument("--medagentbench-judge-max-tokens", type=int, default=512)
    parser.add_argument("--no-reset-medagentbench", action="store_true")
    parser.add_argument("--output-dir", default="eval_results/unified")
    parser.add_argument("--run-id", default=None, help="Reuse a specific run id, primarily for resuming an interrupted run.")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--routed", action="store_true", help="Use the routing layer to select the completion model per episode.")
    parser.add_argument("--router-model-path", default=None, help="Local path to a downloaded bdanko/umsb-distilbert-router model.")
    parser.add_argument("--episodic-memory-model-path", default=None, help="Local sentence-transformers model path for episodic memory retrieval.")
    parser.add_argument("--episodic-memory-dataset-path", default=None, help="Local load_from_disk dataset path with episodic memory chunks.")
    parser.add_argument("--episodic-memory-chunks-path", default=None, help="Local JSONL memory chunks aligned to precomputed embeddings.")
    parser.add_argument("--episodic-memory-embeddings-path", default=None, help="Local .npy matrix of normalized memory embeddings.")
    parser.add_argument("--episodic-memory-top-k", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.reset_medagentbench = not args.no_reset_medagentbench
    args.api_semaphore = threading.BoundedSemaphore(args.max_openrouter_requests)
    args.router_runtime = None
    if args.routed and args.router_model_path:
        args.router_runtime = DistilBertRouterRuntime(args.router_model_path)
    args.episodic_memory_runtime = None
    if args.episodic_memory_model_path and (
        (args.episodic_memory_chunks_path and args.episodic_memory_embeddings_path)
        or args.episodic_memory_dataset_path
    ):
        args.episodic_memory_runtime = EpisodicMemoryRuntime(
            args.episodic_memory_model_path,
            args.episodic_memory_top_k,
            dataset_path=args.episodic_memory_dataset_path,
            chunks_path=args.episodic_memory_chunks_path,
            embeddings_path=args.episodic_memory_embeddings_path,
        )
    if not os.getenv("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required.")
    run_id = args.run_id or now_run_id()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{run_id}_{args.split}_episodes.jsonl"
    summary_path = output_dir / f"{run_id}_{args.split}_summary.json"
    memory_path = output_dir / f"{run_id}_{args.split}_episodic_memory_candidates.jsonl"

    client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=args.api_base)
    judge_client = OpenAI(api_key=os.getenv("OPENROUTER_API_KEY"), base_url=args.api_base)
    ds = load_dataset(args.dataset, split=args.split)
    rows = select_rows((dict(row) for row in ds), args.benchmarks, args.limit, args.limit_per_benchmark)
    done_ids = existing_task_ids(result_path) if args.resume else set()
    results: list[EpisodeResult] = []

    print(f"Run: {run_id}")
    print(f"Dataset: {args.dataset} split={args.split} rows={len(rows)}")
    print(f"Model: {args.model}; judge={args.judge_model}")
    medagentbench_serial = args.reset_medagentbench
    print(f"Parallelism: {args.parallelism} (medagentbench serial={medagentbench_serial})")
    print(f"Max OpenRouter requests: {args.max_openrouter_requests}")
    print(f"Writing: {result_path}")

    pending_rows = [(idx, row) for idx, row in enumerate(rows, start=1) if row["id"] not in done_ids]
    if args.parallelism <= 1:
        for idx, row in pending_rows:
            print(f"[{idx}/{len(rows)}] {row['benchmark']} {row['source_id']}")
            result = run_one(row, client, judge_client, args, run_id)
            record_result(result, result_path, memory_path)
            results.append(result)
            print(f"  reward={result.reward} scorer={result.scorer} error={result.error}")
    else:
        parallel_rows = [
            (idx, row)
            for idx, row in pending_rows
            if row["benchmark"] != "medagentbench" or not medagentbench_serial
        ]
        serial_rows = [
            (idx, row)
            for idx, row in pending_rows
            if row["benchmark"] == "medagentbench" and medagentbench_serial
        ]
        completed = 0
        with ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = {
                executor.submit(run_one, row, client, judge_client, args, run_id): (idx, row)
                for idx, row in parallel_rows
            }
            for future in as_completed(futures):
                idx, row = futures[future]
                result = future.result()
                record_result(result, result_path, memory_path)
                results.append(result)
                completed += 1
                print(
                    f"[{idx}/{len(rows)} complete; parallel {completed}/{len(parallel_rows)}] "
                    f"{row['benchmark']} {row['source_id']} reward={result.reward} scorer={result.scorer} error={result.error}"
                )
        for idx, row in serial_rows:
            print(f"[{idx}/{len(rows)} serial] {row['benchmark']} {row['source_id']}")
            result = run_one(row, client, judge_client, args, run_id)
            record_result(result, result_path, memory_path)
            results.append(result)
            print(f"  reward={result.reward} scorer={result.scorer} error={result.error}")

    all_results = load_episode_results(result_path)
    summary = {
        "run_id": run_id,
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "judge_model": args.judge_model,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "n": len(all_results),
        "by_benchmark": summarize(all_results),
        "usage": summarize_usage(all_results),
        "routing": summarize_routing(all_results),
        "result_path": str(result_path),
        "episodic_memory_candidates_path": str(memory_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
