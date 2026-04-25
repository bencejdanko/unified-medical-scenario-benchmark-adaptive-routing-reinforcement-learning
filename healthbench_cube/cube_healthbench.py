import json
import os
import re
from typing import Dict, Any, List, Optional, Tuple, ClassVar, Type
import blobfile as bf
from pydantic import PrivateAttr, Field
from cube.task import Task, TaskMetadata, TaskConfig
from cube.benchmark import Benchmark, BenchmarkMetadata, RuntimeContext
from cube.tool import ToolboxConfig
from cube.core import Observation, StructuredContent, EnvironmentOutput, Action, Content
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
    _data: Dict[str, Any] = PrivateAttr()
    _grader_client: OpenAI = PrivateAttr()
    _done: bool = PrivateAttr(default=False)

    def __init__(self, data: Dict[str, Any], **kwargs):
        super().__init__(**kwargs)
        self._data = data
        self._done = False
        
        self._grader_client = None

    def reset(self) -> Tuple[Observation, Dict[str, Any]]:
        self._done = False
        # In CUBE, we return an Observation. 
        # For HealthBench, the prompt is already a list of messages.
        obs = Observation(contents=[
            Content.from_data(self._data["prompt"], name="prompt"),
            Content.from_data(self._data["rubrics"], name="rubrics")
        ])
        return obs, {}

    def _grade_response(self, response_text: str) -> float:
        convo_with_response = self._data["prompt"] + [{"role": "assistant", "content": response_text}]
        convo_str = "\n\n".join([f"{m['role']}: {m['content']}" for m in convo_with_response])
        
        rubric_items = self._data["rubrics"]
        total_possible = sum(r["points"] for r in rubric_items if r["points"] > 0)
        if total_possible == 0:
            return 0.0
        
        achieved = 0.0
        if self._grader_client is None:
            clean_url = OPENAI_API_URL
            if clean_url and clean_url.endswith("/chat/completions"):
                clean_url = clean_url[:-len("/chat/completions")]
            
            if not OPENAI_API_KEY:
                raise ValueError("OPENAI_API_KEY is not set. Grading requires an API key.")
                
            self._grader_client = OpenAI(base_url=clean_url, api_key=OPENAI_API_KEY)

        for item in rubric_items:
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

    def evaluate(self, obs: Observation) -> Tuple[float, Dict[str, Any]]:
        # Usually called after a step. 
        # In this task, reward is only meaningful after the 'answer' action.
        return 0.0, {} # Reward is handled in step() for this specific task type

    def step(self, action: Action | List[Action]) -> EnvironmentOutput:
        if self._done:
            raise RuntimeError("Task finished.")
        
        # Action is now a CUBE Action object
        actions = action if isinstance(action, list) else [action]
        last_reward = 0.0
        
        for act in actions:
            if act.name == "answer":
                answer = act.arguments.get("content", "")
                last_reward = self._grade_response(answer)
                self._done = True
                return EnvironmentOutput(
                    obs=Observation.from_text("Task completed."),
                    reward=last_reward,
                    done=True,
                    info={"score": last_reward}
                )
        
        return EnvironmentOutput(
            obs=Observation.from_text("Waiting for answer..."),
            reward=0.0,
            done=False
        )

class HealthBenchTaskConfig(TaskConfig):
    data: Dict[str, Any] = Field(..., description="Task data record")

    def make(self, runtime_context: RuntimeContext | None = None, container_backend: Any = None) -> HealthBenchTask:
        # Get metadata from the benchmark class (if possible) or create it
        # In CUBE, metadata is usually passed in
        # We need to find the metadata for this task_id
        # For simplicity, we assume the benchmark class is globally accessible or we reconstruct it
        
        # Reconstruct metadata (or we could fetch it from a registry)
        from cube.task import TaskMetadata
        metadata = TaskMetadata(id=self.task_id, abstract_description="HealthBench medical rubric task")
        
        return HealthBenchTask(
            data=self.data,
            metadata=metadata,
            tool_config=self.tool_config or ToolboxConfig(),
            runtime_context=runtime_context
        )

class HealthBenchBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        name="HealthBench-Benchmark",
        description="Benchmark for medical rubric evaluation.",
        version="1.0.0",
        extra_info={"id": "healthbench-benchmark-v1"}
    )

    # We can't easily populate 5000 tasks at class level without reading the file
    # CUBE typically expects task_metadata to be a dict[str, TaskMetadata]
    task_metadata: ClassVar[Dict[str, TaskMetadata]] = {}
    task_config_class: ClassVar[Type[TaskConfig]] = HealthBenchTaskConfig

    _task_data: Dict[str, Dict[str, Any]] = PrivateAttr(default_factory=dict)

    def __init__(self, num_examples: Optional[int] = None, **kwargs):
        super().__init__(**kwargs)
        self._load_tasks(num_examples)

    def _setup(self):
        pass

    def close(self):
        pass

    def _load_tasks(self, num_examples: Optional[int]):
        # Clear existing
        self._task_data = {}
        new_metadata = {}
        
        with bf.BlobFile(INPUT_PATH, "rb") as f:
            for i, line in enumerate(f):
                if num_examples is not None and i >= num_examples:
                    break
                data = json.loads(line)
                task_id = f"hb_{i}"
                self._task_data[task_id] = data
                new_metadata[task_id] = TaskMetadata(
                    id=task_id,
                    abstract_description=f"HealthBench task {i}",
                )
        
        # Shadow the class variable with instance data
        object.__setattr__(self, "task_metadata", new_metadata)

    def get_task_configs(self) -> List[HealthBenchTaskConfig]:
        configs = []
        for task_id, tm in self.task_metadata.items():
            configs.append(HealthBenchTaskConfig(
                task_id=task_id,
                data=self._task_data[task_id],
                tool_config=ToolboxConfig()
            ))
        return configs

    def get_task(self, index: int) -> HealthBenchTask:
        # Compatibility with existing test scripts
        task_id = list(self.task_metadata.keys())[index]
        config = HealthBenchTaskConfig(
            task_id=task_id,
            data=self._task_data[task_id],
            tool_config=ToolboxConfig()
        )
        return config.make()

    def __len__(self):
        return len(self.task_metadata)

HealthBenchTask.model_rebuild()
HealthBenchBenchmark.model_rebuild()
HealthBenchTaskConfig.model_rebuild()
