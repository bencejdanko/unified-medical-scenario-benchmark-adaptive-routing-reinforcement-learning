import json
import os
import re
import hashlib
from typing import Dict, Any, List, Optional, Tuple, ClassVar, Type, Literal
import blobfile as bf
from pydantic import PrivateAttr
from cube.task import Task, TaskMetadata
from cube.benchmark import Benchmark, BenchmarkMetadata, TaskConfig, RuntimeContext
from cube.tool import ToolboxConfig
from openai import OpenAI
from dotenv import load_dotenv

# Load credentials
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_URL = os.getenv("OPENAI_API_URL")

INPUT_PATH = "healthbench_cube/data/healthbench.jsonl"

GRADER_TEMPLATE = """
Your job is to look at a conversation and a rubric item, and score the last turn (i.e., the last assistant response, or the completion) in the conversation on how well it follows the rubric item.

# Conversation
<<conversation>>

# Rubric item
<<rubric_item>>

# Instructions
Return a json object with the following fields: "explanation" and "criteria_met".
- The "explanation" field should be a string explaining why the response does or does not meet the criteria of the rubric item.
- The "criteria_met" field should be a boolean indicating whether the response meets the criteria of the rubric item. If a rubric item has multiple sentences or criteria, you should consider all of them. If any of the criteria is not met, the answer should be false. Only return true is all of the criteria are met.
- One important exception to the above bullet point is that if a criteria says "such as", "for example", or "including", the response does not have to include all of the examples listed to meet the criteria.

# Final instruction
Return just the json object in markdown format. Do not include any other text in the response.
""".strip()

def parse_json_to_dict(json_string: str) -> dict:
    json_cleaned = re.sub(r"^```json\s*|\s*```$", "", json_string.strip())
    try:
        return json.loads(json_cleaned)
    except json.JSONDecodeError:
        return {}

class HealthBenchTask(Task):
    task_metadata: ClassVar[TaskMetadata] = TaskMetadata(
        id="healthbench-v1",
        name="HealthBench-Task",
        description="A medical rubric-based evaluation task.",
        version="1.0.0"
    )

    _data: Dict[str, Any] = PrivateAttr()
    _task_id: str = PrivateAttr()
    _done: bool = PrivateAttr(default=False)
    _grader_client: OpenAI = PrivateAttr()

    def __init__(self, data: Dict[str, Any], task_id: str, **kwargs):
        if "metadata" not in kwargs:
            kwargs["metadata"] = self.task_metadata
        if "tool_config" not in kwargs:
            kwargs["tool_config"] = ToolboxConfig()
        super().__init__(**kwargs)
        self._data = data
        self._task_id = task_id
        self._done = False
        
        clean_url = OPENAI_API_URL
        if clean_url and clean_url.endswith("/chat/completions"):
            clean_url = clean_url[:-len("/chat/completions")]
        self._grader_client = OpenAI(base_url=clean_url, api_key=OPENAI_API_KEY)

    def reset(self, config: Optional[TaskConfig] = None) -> Dict[str, Any]:
        self._done = False
        return {
            "prompt": self._data["prompt"],
            "rubrics": self._data["rubrics"]
        }

    def _grade_response(self, response_text: str) -> float:
        convo_with_response = self._data["prompt"] + [{"role": "assistant", "content": response_text}]
        convo_str = "\n\n".join([f"{m['role']}: {m['content']}" for m in convo_with_response])
        
        rubric_items = self._data["rubrics"]
        total_possible = sum(r["points"] for r in rubric_items if r["points"] > 0)
        if total_possible == 0:
            return 0.0
        
        achieved = 0.0
        for item in rubric_items:
            # Simple grading call
            prompt = GRADER_TEMPLATE.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", item["criterion"])
            try:
                res = self._grader_client.chat.completions.create(
                    model="openai/gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}]
                )
                grading = parse_json_to_dict(res.choices[0].message.content)
                if grading.get("criteria_met"):
                    achieved += item["points"]
            except Exception as e:
                print(f"Grading error: {e}")
        
        return achieved / total_possible

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Task finished.")
        
        if action.get("type") == "answer":
            answer = action.get("content", "")
            reward = self._grade_response(answer)
            self._done = True
            return {}, reward, True, {"score": reward}
        
        return {"status": "error", "message": "Invalid action"}, 0.0, False, {}

    def evaluate(self) -> Dict[str, Any]:
        return {"task_id": self._task_id, "done": self._done}

class HealthBenchBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        id="healthbench-benchmark-v1",
        name="HealthBench-Benchmark",
        description="Benchmark for medical rubric evaluation.",
        version="1.0.0"
    )

    # CUBE protocol requirements
    task_metadata: ClassVar[Dict[str, Any]] = {
        "id": "healthbench-v1",
        "name": "HealthBench-Task",
        "description": "A medical rubric-based evaluation task.",
        "version": "1.0.0"
    }
    
    task_config_class: ClassVar[Type[TaskConfig]] = TaskConfig

    _tasks: List[HealthBenchTask] = PrivateAttr(default_factory=list)
    _num_examples: Optional[int] = PrivateAttr()

    def __init__(self, num_examples: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._num_examples = num_examples
        self._tasks = []

    def _setup(self):
        self._load_tasks(self._num_examples)

    def close(self):
        pass

    def _load_tasks(self, num_examples: Optional[int]):
        with bf.BlobFile(INPUT_PATH, "rb") as f:
            for i, line in enumerate(f):
                if num_examples is not None and i >= num_examples:
                    break
                data = json.loads(line)
                self._tasks.append(HealthBenchTask(data, f"hb_{i}"))

    def get_task(self, index: int, config: Optional[TaskConfig] = None) -> HealthBenchTask:
        if not self._tasks:
            self._setup()
        return self._tasks[index]

    def __len__(self):
        if not self._tasks:
            self._setup()
        return len(self._tasks)

HealthBenchTask.model_rebuild()
HealthBenchBenchmark.model_rebuild()
