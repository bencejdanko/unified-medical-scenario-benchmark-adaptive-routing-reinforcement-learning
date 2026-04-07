import json
from typing import Dict, Any, List, Optional, Tuple, ClassVar, Type
from cube.task import Task, TaskMetadata
from cube.benchmark import Benchmark, BenchmarkMetadata, TaskConfig, RuntimeContext
from cube.tool import ToolboxConfig
from pydantic import PrivateAttr

# MedQA Task for CUBE
# This wraps a medical question into the CUBE standard interaction loop.

class MedQATask(Task):
    # Task class expects TaskMetadata instance
    task_metadata: ClassVar[TaskMetadata] = TaskMetadata(
        id="medqa-usmle-v1",
        name="MedQA-USMLE-Task",
        description="A single USMLE medical question answering task.",
        version="1.0.0"
    )
    
    # Private attributes that won't be validated by Pydantic
    _data: Dict[str, Any] = PrivateAttr()
    _task_id: str = PrivateAttr()
    _done: bool = PrivateAttr(default=False)
    _history: List[Dict[str, Any]] = PrivateAttr(default_factory=list)

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
        """Reset the task and return the initial observation."""
        self._done = False
        self._history = []
        return {
            "question": self._data["question"],
            "options": self._data["options"],
            "context": self._data.get("context", ""),
            "metadata": self._data.get("metadata", {})
        }

    def step(self, action: Dict[str, Any]) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        if self._done:
            raise RuntimeError("Task is already finished.")

        action_type = action.get("type")
        ans_target = self._data.get("answer_idx")
        
        if action_type == "answer":
            prediction = action.get("idx")
            reward = 1.0 if str(prediction) == str(ans_target) else 0.0
            self._done = True
            return {}, reward, True, {"correct": str(prediction) == str(ans_target)}
        elif action_type == "tool_call":
            self._history.append(action)
            return {"status": "tool_called", "action": action}, 0.0, False, {}
        else:
            raise ValueError(f"Invalid action type: {action_type}")

    def evaluate(self) -> Dict[str, Any]:
        return {
            "task_id": self._task_id,
            "done": self._done,
            "history_length": len(self._history)
        }

class MedQABenchmark(Benchmark):
    benchmark_metadata: ClassVar[BenchmarkMetadata] = BenchmarkMetadata(
        id="medqa-benchmark-v1",
        name="MedQA-USMLE-Benchmark",
        description="Benchmark collection for Medical Question Answering (MedQA/USMLE).",
        version="1.0.0"
    )
    
    # CUBE protocol requirements
    task_metadata: ClassVar[Dict[str, Any]] = {
        "id": "medqa-usmle-v1",
        "name": "MedQA-USMLE-Task",
        "description": "A single USMLE medical question answering task.",
        "version": "1.0.0"
    }
    
    task_config_class: ClassVar[Type[TaskConfig]] = TaskConfig

    _data_path: str = PrivateAttr()
    _tasks_list: List[MedQATask] = PrivateAttr(default_factory=list)

    def __init__(self, data_path: str, **kwargs):
        super().__init__(**kwargs)
        self._data_path = data_path
        self._tasks_list = []

    def _setup(self):
        self._load_tasks()

    def _load_tasks(self):
        with open(self._data_path, "r") as f:
            for i, line in enumerate(f):
                data = json.loads(line)
                self._tasks_list.append(MedQATask(data, f"medqa_{i}"))

    def get_task(self, index: int, config: Optional[TaskConfig] = None) -> MedQATask:
        if not self._tasks_list:
            self._setup()
        return self._tasks_list[index]

    def close(self):
        pass

    def __len__(self):
        if not self._tasks_list:
            self._setup()
        return len(self._tasks_list)

# Fix for Pydantic early binding/rebuilding
MedQATask.model_rebuild()
MedQABenchmark.model_rebuild()
