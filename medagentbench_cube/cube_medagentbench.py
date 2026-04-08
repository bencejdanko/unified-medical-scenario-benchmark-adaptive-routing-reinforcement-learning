import json
import os
import requests
from typing import Dict, Any, List, Optional, Tuple, ClassVar, Type
from pydantic import PrivateAttr
from cube.task import Task, TaskMetadata
from cube.benchmark import Benchmark, BenchmarkMetadata, TaskConfig, RuntimeContext
from cube.tool import ToolboxConfig
from dotenv import load_dotenv

load_dotenv()

class MedAgentBenchTask(Task):
    task_metadata: ClassVar[TaskMetadata] = TaskMetadata(
        id="medagentbench-v1",
        name="MedAgentBench-Task",
        description="A medical agent task involving tool calls to an EHR.",
        version="1.0.0"
    )

    _data: Dict[str, Any] = PrivateAttr()
    _task_id: str = PrivateAttr()
    _done: bool = PrivateAttr(default=False)
    _history: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _fhir_api_base: str = PrivateAttr(default="http://localhost:8080/api/")

    def __init__(self, data: Dict[str, Any], task_id: str, **kwargs):
        if "metadata" not in kwargs:
            kwargs["metadata"] = self.task_metadata
        if "tool_config" not in kwargs:
            kwargs["tool_config"] = ToolboxConfig()
        super().__init__(**kwargs)
        self._data = data
        self._task_id = task_id
        self._done = False
        self._history = []

    def reset(self, config: Optional[TaskConfig] = None) -> Dict[str, Any]:
        self._done = False
        self._history = []
        return {
            "instruction": self._data["instruction"],
            "context": self._data.get("context", "")
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Task finished")
        
        self._history.append(action)
        action_type = action.get("type")
        
        if action_type == "tool_call":
            tool_name = action.get("name")
            parameters = action.get("parameters", {})
            try:
                # Strip api_base placeholder if present
                path = tool_name.replace("GET ", "").replace("POST ", "").replace("{api_base}", "").strip()
                method = "GET" if "GET" in tool_name else "POST"
                url = self._fhir_api_base.rstrip("/") + path
                
                if method == "GET":
                    res = requests.get(url, params=parameters, timeout=10)
                else:
                    res = requests.post(url, json=parameters, timeout=10)
                
                if res.status_code == 200:
                    observation = res.json()
                else:
                    observation = {"error": f"HTTP {res.status_code}: {res.text}"}
            except Exception as e:
                observation = {"error": f"Failed to connect to FHIR server: {e}"}
            
            return {"observation": observation}, 0.0, False, {}

        elif action_type == "answer":
            prediction = action.get("content", "")
            sol = self._data.get("sol", [])
            is_correct = False
            if isinstance(sol, list):
                is_correct = any(str(prediction).strip().lower() == str(s).strip().lower() for s in sol)
            else:
                is_correct = str(prediction).strip().lower() == str(sol).strip().lower()
            
            reward = 1.0 if is_correct else 0.0
            self._done = True
            return {}, reward, True, {"correct": is_correct}

        return {"error": "Invalid action type"}, 0.0, False, {}

    def evaluate(self) -> Dict[str, Any]:
        return {"task_id": self._task_id, "done": self._done, "history": self._history}

class MedAgentBenchBenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        id="medagentbench-benchmark-v1",
        name="MedAgentBench-Benchmark",
        description="Benchmark for medical agents (EHR tasks).",
        version="1.0.0"
    )
    
    task_metadata: ClassVar[Dict[str, Any]] = {
        "id": "medagentbench-v1",
        "name": "MedAgentBench-Task",
        "description": "A medical agent task involving tool calls to an EHR.",
        "version": "1.0.0"
    }
    
    task_config_class: ClassVar[Type[TaskConfig]] = TaskConfig

    _data_path: str = PrivateAttr()
    _tasks: List[MedAgentBenchTask] = PrivateAttr(default_factory=list)

    def __init__(self, data_path: str, **kwargs):
        super().__init__(**kwargs)
        self._data_path = data_path
        self._tasks = []

    def _setup(self):
        with open(self._data_path, "r") as f:
            data_list = json.load(f)
            for i, data in enumerate(data_list):
                self._tasks.append(MedAgentBenchTask(data, f"mb_{i}"))

    def get_task(self, index: int, config: Optional[TaskConfig] = None) -> MedAgentBenchTask:
        if not self._tasks:
            self._setup()
        return self._tasks[index]

    def __len__(self):
        if not self._tasks:
            self._setup()
        return len(self._tasks)

    def close(self):
        pass

MedAgentBenchTask.model_rebuild()
MedAgentBenchBenchmark.model_rebuild()
